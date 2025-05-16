import os
import onnx
import torch
import logging
import argparse
import onnxruntime as ort
from core.config import cfg
from core.utils import dist_util
from core.utils.logger import setup_logger
from core.modeling.model.stad import build_stad
from core.utils.checkpoint import CheckPointer


def load_model(cfg):
    # create device
    device = torch.device(cfg.MODEL.DEVICE)

    # create model
    model = build_stad(cfg)
    model = model.to(device)
    model.eval()

    # load weights
    checkpointer = CheckPointer(model, None, None, cfg.OUTPUT_DIR)
    checkpointer.load()

    return model


def main() -> int:
    # Create argument parser
    parser = argparse.ArgumentParser(description='PyTorch Export To ONNX')
    parser.add_argument('--config-file', dest='config_file', type=str, default="outputs/train_ucf_2/cfg.yaml",
                        help="Path to config file")
    parser.add_argument('--onnx-opset', dest="onnx_opset", type=int, default=12,
                        help='Target onnx opset')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()

    # enable cudnn auto-tuner
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # read config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # create logger
    logger = setup_logger("EXPORT", dist_util.get_rank())
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    # create output export dir
    folder_path = os.path.join(cfg.OUTPUT_DIR, "export/onnx/")
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    # export model
    with torch.no_grad():
        # prepare result filenames
        basename = "stad_" + cfg.MODEL.BACKBONE2D.ARCHITECTURE + "_" + cfg.MODEL.BACKBONE3D.ARCHITECTURE
        model_filename = os.path.join(folder_path, basename + ".onnx")
        model_imp_filename = os.path.join(folder_path, basename + "_imp" + ".onnx")

        # load model
        torch.cuda.empty_cache()
        device = torch.device(cfg.MODEL.DEVICE)
        model = load_model(cfg)

        # export to onnx
        w, h = cfg.INPUT.IMAGE_SIZE
        t = cfg.DATASET.SEQUENCE_LENGTH
        in_clip = torch.randn(size=(1, 3, t, h, w), dtype=torch.float32).to(device) # (B, C, T, H, W)
        in_keyframe = torch.randn(size=(1, 3, h, w), dtype=torch.float32).to(device) # (B, C, H, W)

        torch.onnx.export(model,
                          (in_clip, in_keyframe),
                          model_filename,
                          verbose=False, 
                          opset_version=args.onnx_opset,
                          keep_initializers_as_inputs = False,
                          input_names=["in_clip", "in_keyframe"],
                          output_names=["out_y", "out_x"]
        )

        # validate model
        onnx_model = onnx.load(model_filename)
        onnx.checker.check_model(onnx_model)

        # export imp model
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = model_imp_filename
        session = ort.InferenceSession(model_filename, so,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    return 0


if __name__ == '__main__':
    main()
