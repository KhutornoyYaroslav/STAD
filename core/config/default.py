from yacs.config import CfgNode as CN

_CFG = CN()

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_CFG.MODEL = CN()
_CFG.MODEL.DEVICE = "cuda"
_CFG.MODEL.PRETRAINED_WEIGHTS = ""

# ---------------------------------------------------------------------------- #
# Backbone2d
# ---------------------------------------------------------------------------- #
_CFG.MODEL.BACKBONE2D = CN()
_CFG.MODEL.BACKBONE2D.ARCHITECTURE = "yolov8m"
_CFG.MODEL.BACKBONE2D.PRETRAINED_WEIGHTS = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
_CFG.MODEL.BACKBONE2D.FREEZE = False

# ---------------------------------------------------------------------------- #
# Backbone3d
# ---------------------------------------------------------------------------- #
_CFG.MODEL.BACKBONE3D = CN()
_CFG.MODEL.BACKBONE3D.ARCHITECTURE = "mobilenetv2"
_CFG.MODEL.BACKBONE3D.CHANNEL_MULT = 1.0
_CFG.MODEL.BACKBONE3D.PRETRAINED_WEIGHTS = ""
_CFG.MODEL.BACKBONE3D.FREEZE = False

# ---------------------------------------------------------------------------- #
# FeatureFusion
# ---------------------------------------------------------------------------- #
_CFG.MODEL.FEATURE_FUSION = CN()
_CFG.MODEL.FEATURE_FUSION.INTER_CHANNELS = 256

# ---------------------------------------------------------------------------- #
# Head
# ---------------------------------------------------------------------------- #
_CFG.MODEL.HEAD = CN()
_CFG.MODEL.HEAD.ARCHITECTURE = "yolov8"
_CFG.MODEL.HEAD.PRETRAINED_WEIGHTS = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
_CFG.MODEL.HEAD.FREEZE = False
_CFG.MODEL.HEAD.NUM_CLASSES = 80

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
_CFG.LOSS = CN()
_CFG.LOSS.BOX_WEIGHT = 1.0
_CFG.LOSS.CLS_WEIGHT = 1.0
_CFG.LOSS.DFL_WEIGHT = 1.0
_CFG.LOSS.TAL_TOPK = 10

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_CFG.INPUT = CN()
_CFG.INPUT.IMAGE_SIZE = [512, 512]
_CFG.INPUT.PAD_LABELS_TO = 32
_CFG.INPUT.PIXEL_MEAN = [0, 0, 0]
_CFG.INPUT.PIXEL_SCALE = [255, 255, 255]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_CFG.DATASET = CN()
_CFG.DATASET.TRAIN_DATA_PATHS = []
_CFG.DATASET.TRAIN_ANNO_PATHS = []
_CFG.DATASET.VALID_DATA_PATHS = []
_CFG.DATASET.VALID_ANNO_PATHS = []
_CFG.DATASET.CLASS_LABELS_FILE = ""
_CFG.DATASET.TYPE = ""
_CFG.DATASET.SEQUENCE_LENGTH = 1
_CFG.DATASET.SEQUENCE_STRIDE = 1
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_CFG.DATA_LOADER = CN()
_CFG.DATA_LOADER.NUM_WORKERS = 1
_CFG.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_CFG.SOLVER = CN()
_CFG.SOLVER.BATCH_SIZE = 1
_CFG.SOLVER.LR = 1e-4
_CFG.SOLVER.MAX_EPOCH = 10
_CFG.SOLVER.WEIGHT_DECAY = 0.0

# ---------------------------------------------------------------------------- #
# Output options
# ---------------------------------------------------------------------------- #
_CFG.OUTPUT_DIR = 'outputs/test'

# ---------------------------------------------------------------------------- #
# Tensorboard
# ---------------------------------------------------------------------------- #
_CFG.TENSORBOARD = CN()
_CFG.TENSORBOARD.BEST_SAMPLES_NUM = 32
_CFG.TENSORBOARD.CONF_THRESH = 0.5
_CFG.TENSORBOARD.WORST_SAMPLES_NUM = 32
