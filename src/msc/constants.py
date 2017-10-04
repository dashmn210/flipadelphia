import sys
sys.path.append('../..')

from src.models.dummies.tf_utils import TFModelWrapper


MODEL_CLASSES = {
    'neural': TFModelWrapper
}
