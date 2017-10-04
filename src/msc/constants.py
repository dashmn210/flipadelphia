import sys
sys.path.append('../..')

from src.models.neural.tf_utils import TFModelWrapper


MODEL_CLASSES = {
    'neural': TFModelWrapper
}
