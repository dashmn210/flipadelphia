import sys
sys.path.append('../..')

from src.models.dummies.tf_dummy import TFDummyWrapper


MODEL_CLASSES = {
    'neural': TFDummyWrapper
}
