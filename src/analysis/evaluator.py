import numpy as np
import sys
sys.path.append('../..')
import src.msc.utils as utils
from correlations import cramers_v



def evaluate(config, dataset, predictions, model_dir):
    # predictions = abstract_model.Prediction
    performance = {}
    correlations = {}

    for var in config.data_spec[1:]:
        if var['control']:
            features = dataset.features
            input_text = dataset.get_tokenized_input()
            labels = dataset.data_for_var(var)
            importance_threshold = utils.percentile(
                predictions.feature_importance.values(),
                config.feature_importance_threshold)
            print features
            print predictions.feature_importance.keys()

            correlations[var] = np.mean([
                cramers_v(
                    feature=f, 
                    text=input_text, 
                    num_levels=dataset.num_levels(var['name']),
                    labels=labels) \
                for f in features \
                if predictions.feature_importance[f] > importance_threshold
            ])


        # if in scores...



