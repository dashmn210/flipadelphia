import sys
sys.path.append('../..')

from src.models.neural.tf_wrapper import TFFlipperWrapper
from src.models.neural.tf_wrapper import TFStackedRegressionWrapper

from src.models.linear.mixed_regression import MixedRegression
from src.models.linear.plain_regression import RegularizedRegression
from src.models.linear.fixed_regression import FixedRegression



MODEL_CLASSES = {
    'neural': TFFlipperWrapper,
    'stacked-regression': TFStackedRegressionWrapper,
    'mixed-regression': MixedRegression,
    'fixed-regression': FixedRegression,
    'regression': RegularizedRegression,
}
