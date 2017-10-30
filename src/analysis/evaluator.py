import numpy as np
import sys
sys.path.append('../..')
import src.msc.utils as utils
from src.models.linear.plain_regression import RegularizedRegression

from correlations import cramers_v, pointwise_biserial
import sklearn.metrics
import os
import time

def write_summary(evaluation, model_dir):
    """ evaluation: product of evaluator.evaluate()
    """
    bar = '=' * 40
    s = bar + '\n'
    s += 'PERFORMANCE:\n'
    for response, metrics in evaluation['performance'].items():
        s += '%s:\n' % response
        if isinstance(metrics, dict):
            s += '\n'.join('%s:\t%s' % (metric, val) for metric, val in metrics.items()) + '\n\n'
        else: 
            s += '%s\n\n' % metrics
    s += bar + '\n\n\n\n'
    s += 'CORRELATIONS:\n'
    s += '\n'.join('%s:\t%s' % (k, v) for k, v in evaluation['correlations'].items()) + '\n\n'
    s += bar + '\n'

    with open(os.path.join(model_dir, 'summary.txt'), 'w') as f:
        f.write(s)


def feature_correlation(var, dataset, features, input_text, labels):
    """ compute the correlation between a list of features and the
        labels for a particular variable (var)
    """
    if var['type'] == 'categorical':
        return np.mean([
            cramers_v(
                feature=f, 
                text=input_text, 
                targets=labels,
                possible_labels=list(set(labels))) \
            for f in features
        ])
    else:
        return np.mean([
            pointwise_biserial(
                feature=f,
                text=input_text,
                targets=labels) \
            for f in features \
        ])


def eval_performance(var, labels, preds, dataset):
    """ evaluate the performance of some predictions
        return {metric: value}
    """
    if np.all(np.isnan(preds)):
        return 'N/A (all nans)'

    elif var['type'] == 'categorical':
        # replace labels with their ids
        labels = map(
            lambda x: dataset.class_to_id_map[var['name']][x],
            labels)
        labels_hat = np.argmax(preds, axis=1)
        report = sklearn.metrics.classification_report(labels, labels_hat)
        # TODO verify that cols of preds are ordered correctly for sklearn
        xentropy = sklearn.metrics.log_loss(labels, preds)
        return {'report': report, 'xenropy': xentropy}

    else:
        # filter out nans
        labels, preds = zip(*[
            (label, pred) for label, pred in zip(labels, preds) \
            if not np.isnan(pred)])
        MSE = sklearn.metrics.mean_squared_error(labels, preds)
        r2 = sklearn.metrics.r2_score(labels, preds)
        return { 'MSE': MSE, 'R^2': r2 }


def evaluate(config, dataset, predictions, model_dir):
    """ config: config object
        dataset: data.dataset object
        predictions: models.abstract_model.Prediction
        model_dir: output dir

        produces evaluation metrics
            - feature correlations with confounds
            - target variable prediction qualtiy
    """
    performance = {}
    correlations = {}

    # select features
    importance_threshold = utils.percentile(
        predictions.feature_importance.values(),
        config.feature_importance_threshold)
    features = [
        f for f in dataset.features \
        if predictions.feature_importance.get(f, 0) > importance_threshold
    ]

    # use these selected features to train & test a new model
    # TODO -- make batch size etc config params
    print 'EVALUATOR: running linear model with selected features'
    m = RegularizedRegression(config, {'batch_size': 8, 'num_train_steps': 1000}, intercept=False)
    dataset.set_active_split(config.train_suffix)
    m.train(dataset, '', features=features)
    dataset.set_active_split(config.test_suffix)
    feature_predictions = m.inference(dataset, '')

    # now evaluate the selected features, both in terms of
    #  correlation with confounds and ability to predict response
    for var in config.data_spec[1:]:
        if var['skip']: continue

        labels = dataset.data_for_var(var)

        if var['control']:
            start = time.time()
            print 'EVALUATOR: computing %s correlation...' % var['name']
            correlations[var['name']] = feature_correlation(
                var=var,
                features=features,
                input_text=dataset.get_tokenized_input(),
                labels=labels,
                dataset=dataset)
            print '\t Done. took %.2fs' % (time.time() - start)

        else:
            assert var['name'] in feature_predictions.scores
            preds = feature_predictions.scores[var['name']]
            assert len(preds) == len(labels)

            performance[var['name']] = eval_performance(
                var=var,
                labels=labels,
                preds=preds,
                dataset=dataset)

    return {'correlations': correlations, 'performance': performance}


