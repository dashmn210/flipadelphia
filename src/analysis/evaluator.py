import numpy as np
import sys
sys.path.append('../..')
import src.msc.utils as utils
from correlations import cramers_v, pointwise_biserial
import sklearn.metrics
import os


def write_summary(evaluation, model_dir):
    """ evaluation: product of evaluator.evaluate()
    """
    bar = '=' * 40
    s = bar + '\n'
    s += 'PERFORMANCE:\n'
    for response, metrics in evaluation['performance'].items():
        s += '%s:\n' % response
        if isinstance(metrics, dict):
            s += '\n'.join('%s:\t%s' % (metric, val) for metric, val in metrics.items()) + '\n'
        else: 
            s += '%s\n' % metrics
    s += bar + '\n'
    s += 'CORRELATIONS:\n'
    s += '\n'.join('%s:\t%s' % (k, v) for k, v in evaluation['correlations'].items()) + '\n'
    with open(os.path.join(model_dir, 'summary'), 'w') as f:
        f.write(s)


def evaluate(config, dataset, predictions, model_dir):
    # TODO -- some kind of plotting?
    # TODO -- refactor big time

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
                correlations[var['name']] = np.mean([
                    pointwise_biserial(
                        feature=f,
                        text=input_text,
                        targets=labels) \
                    for f in features \
                    if predictions.feature_importance.get(f, 0) > importance_threshold
                ])

        else:
            assert var['name'] in predictions.scores
            preds = predictions.scores[var['name']]
            labels = dataset.data_for_var(var)
            assert len(preds) == len(labels)

            if np.all(np.isnan(preds)):
                performance[var['name']] = 'N/A (all nans)'

            elif var['type'] == 'categorical':
                # replace labels with their ids
                labels = map(
                    lambda x: dataset.class_to_id_map[var['name']][x],
                    labels)
                labels_hat = np.argmax(preds, axis=1)
                report = sklearn.metrics.classification_report(labels, labels_hat)
                # TODO verify that cols of preds are ordered correctly for sklearn
                xentropy = sklearn.metrics.log_loss(labels, preds)
                performance[var['name']] = {'report': report, 'xenropy': xentropy}

            else:
                # filter out nans
                labels, preds = zip(*[
                    (label, pred) for label, pred in zip(labels, preds) \
                    if not np.isnan(pred)])
                MSE = sklearn.metrics.mean_squared_error(labels, preds)
                r2 = sklearn.metrics.r2_score(labels, preds)
                performance[var['name']] = { 'MSE': MSE, 'R^2': r2 }

    return {'correlations': correlations, 'performance': performance}


