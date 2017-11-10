import sys
sys.path.append('../..')

from src.models.neural.tf_wrapper import TFFlipperWrapper
from src.models.neural.tf_wrapper import TFBOWFlipperWrapper
from src.models.neural.tf_wrapper import TFCausalRegressionWrapper
from src.models.neural.tf_wrapper import TFCausalWrapper

from src.models.linear.mixed_regression import MixedRegression
from src.models.linear.plain_regression import RegularizedRegression
from src.models.linear.fixed_regression import FixedRegression
from src.models.linear.double_regression import DoubleRegression


MODEL_CLASSES = {
    'neural': TFFlipperWrapper,
    'bow-neural': TFBOWFlipperWrapper,
    'causal-neural': TFCausalWrapper,
    'causal-regression': TFCausalRegressionWrapper,
    'mixed-regression': MixedRegression,
    'fixed-regression': FixedRegression,
    'regression': RegularizedRegression,
    'double-regression': DoubleRegression,
}
