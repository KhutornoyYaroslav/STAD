import os
import onnx
import torch
import argparse
import onnxruntime as ort
from torch import nn
from core.config import cfg
from core.utils import dist_util
from core.modeling import build_model
from core.utils.logger import setup_logger
from core.utils.checkpoint import CheckPointer
from onnxconverter_common import float16


def prepare_model(cfg, device: torch.device) -> nn.Module:
    # create model
    model = build_model(cfg)
    model = model.to(device)
    model = model.eval()
    # model.prepare_inference(*cfg.INPUT.IMAGE_SIZE)

    # load weights
    checkpointer = CheckPointer(model, None, None, cfg.OUTPUT_DIR)
    checkpointer.load()

    return model


def main() -> int:
    # parse arguments
    parser = argparse.ArgumentParser(description='PyTorch Export To ONNX')
    parser.add_argument('--config-file', dest='config_file', type=str, default="outputs/yolov8m_640_384_cls01/cfg.yaml",
                        help="Path to config file")
    parser.add_argument('--name', dest='name', type=str, default="pedact",
                        help="Basename of the model file")
    parser.add_argument('--version', dest='version', type=int, default=0,
                        help="Version of the model")
    parser.add_argument('--person-height', dest="person_height", type=int, default=80,
                        help='Typical height of person bound box at the training stage')
    parser.add_argument('--onnx-opset', dest="onnx_opset", type=int, default=12,
                        help='Target onnx opset')
    args = parser.parse_args()

    # enable cudnn auto-tuner
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # read config
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # create logger
    logger = setup_logger("EXPORT", dist_util.get_rank())
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    # create output export dir
    output_dir = os.path.join(cfg.OUTPUT_DIR, "export/onnx/")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # export model
    with torch.no_grad():
        # prepare output filenames
        basename = args.name + "_v" + str(args.version)
        model_filename = os.path.join(output_dir, basename + ".onnx")
        model_imp_filename = os.path.join(output_dir, basename + "_imp.onnx")
        model_fp16_filename = os.path.join(output_dir, basename + "_fp16.onnx")

        # prepare model
        torch.cuda.empty_cache()
        device = torch.device(cfg.MODEL.DEVICE)
        model = prepare_model(cfg, device)

        # export to onnx
        w, h = cfg.INPUT.IMAGE_SIZE
        t = 1 # cfg.DATASET.SEQUENCE_LENGTH
        input = torch.randn(size=(1, t, 3, h, w), dtype=torch.float32).to(device) # (B, T, C, H, W)

        torch.onnx.export(
            model,
            (input),
            model_filename,
            verbose=False, 
            opset_version=args.onnx_opset,
            keep_initializers_as_inputs=False,
            input_names=["input"],
            output_names=["out_y", "out_x"],
            dynamic_axes={
                "input": {
                    0: "batch", 
                    3: "height",
                    4: "width"
                    },
                "out_x": {0: "batch"},
                "out_y": {0: "batch", 2: "anchors"}
                }
            )

        # validate model
        onnx_model = onnx.load(model_filename)
        onnx.checker.check_model(onnx_model)

        # add custom metadata
        onnx_model.model_version = args.version

        metadata = {
            "labels": str(dict(enumerate(cfg.DATASET.LABELS))),
            "person_typical_height_pix": str(args.person_height),
            "input_color": str(cfg.INPUT.COLOR),
            "input_norm_mean": str(cfg.INPUT.PIXEL_MEAN),
            "input_norm_scale": str(cfg.INPUT.PIXEL_SCALE),
            "input_divisible_by": str(cfg.INPUT.MAKE_DIVISIBLE_BY),
            "padding_border_value": str(cfg.INPUT.PAD_BORDER_VALUE),
            "model_backbone_arch": str(cfg.MODEL.BACKBONE2D.ARCHITECTURE),
        }

        for key, value in metadata.items():
            meta = onnx_model.metadata_props.add()
            meta.key = key
            meta.value = value

        onnx.save(onnx_model, model_filename)

        # export imp model
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = model_imp_filename
        session = ort.InferenceSession(model_filename, so,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # export to float16 model
        onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_model_fp16, model_fp16_filename)

    return 0


if __name__ == '__main__':
    main()
