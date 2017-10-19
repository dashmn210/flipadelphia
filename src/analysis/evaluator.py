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
            if var['type'] == 'categorical':
                correlations[var['name']] = np.mean([
                    cramers_v(
                        feature=f, 
                        text=input_text, 
                        targets=labels,
                        possible_labels=dataset.class_to_id_map[var['name']].keys()) \
                    for f in features \
                    if predictions.feature_importance.get(f, 0) > importance_threshold
                ])
            else:
                # TODO pointwise-biserial
                pass

        # if in scores...
    print correlations
    print performance


