import sys
sys.path.append('../..')

from src.models.neural.tf_wrapper import TFModelWrapper
from src.models.mixed_regression.mixed_model_base import MixedWrapper


MODEL_CLASSES = {
    'neural': TFModelWrapper,
    'mixed-regression': MixedWrapper
}
