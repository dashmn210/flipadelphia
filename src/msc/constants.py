import sys
sys.path.append('../..')

from src.models.neural.tf_wrapper import TFModelWrapper
from src.models.linear.mixed_regression import MixedRegression


MODEL_CLASSES = {
    'neural': TFModelWrapper,
    'mixed-regression': MixedRegression
}
