import numpy as np
import sys
sys.path.append('../..')
import src.msc.utils as utils
from src.models.linear.plain_regression import RegularizedRegression
from src.models.linear.fixed_regression import FixedRegression

from correlations import cramers_v, pointwise_biserial
import sklearn.metrics
import os
import time

def write_summary(evaluation, model_dir):
    """ evaluation: product of evaluator.evaluate()
        TODO -- make more human readable lol
    """
    with open(os.path.join(model_dir, 'summary.txt'), 'w') as f:
        f.write(str(evaluation))


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


def eval_performance(var, labels, regression_preds, fixed_preds, dataset):
    """ evaluate the performance of some predictions
        return {metric: value}
    """
    def eval_categorical(preds, label_ids):
        labels_hat = np.argmax(preds, axis=1)
        report = sklearn.metrics.classification_report(labels, labels_hat)
        f1 = sklearn.metrics.f1_score(labels, labels_hat, average='weighted')
        # TODO verify that cols of preds are ordered correctly for sklearn
        xentropy = sklearn.metrics.log_loss(labels, preds,
            labels=sorted(dataset.id_to_class_map[var['name']].keys()))
        return {'report': report, 'xenropy': xentropy, 'stat': f1}

    def eval_continuous(preds, labels):
        MSE = sklearn.metrics.mean_squared_error(labels, preds)
        r2 = sklearn.metrics.r2_score(labels, preds)
        return { 'MSE': MSE, 'R^2': r2, 'stat': MSE}


    if np.all(np.isnan(regression_preds)) or np.all(np.isnan(fixed_preds)):
        return 'N/A (all nans)'

    elif var['type'] == 'categorical':
        # replace labels with their ids
        labels = map(
            lambda x: dataset.class_to_id_map[var['name']][x],
            labels)
        reg_eval = eval_categorical(regression_preds, labels)
        fixed_eval = eval_categorical(fixed_preds, labels)

    else:
        # filter out nans
        labels, regression_preds, fixed_preds = zip(*[
            (label, p1, p2) for label, p1, p2 in zip(labels, regression_preds, fixed_preds) \
            if not np.isnan(p1) and not np.isnan(p2)])
        reg_eval = eval_continuous(regression_preds, labels)
        fixed_eval = eval_continuous(fixed_preds, labels)
    return {'regression': reg_eval, 'fixed': fixed_eval}


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
    # select K largest features that are in the dataset
    #   (i.e. ignoring fixed features, intercepts, etc)
    features = sorted(
        predictions.feature_importance.items(),
        key=lambda x: x[1])[::-1][:config.num_eval_features]
    features = [x[0] for x in features if x[0] in dataset.features]

    print 'EVALUATOR: writing selected features + weights...'
    s = ''
    for f in features:
        s += '%s\t%.4f\n' % (f, predictions.feature_importance[f])
    with open(os.path.join(model_dir, 'selected-words-and-importance.txt'), 'w') as f:
        f.write(s)

    # use these selected features to train & test a new model
    # TODO -- make batch size etc config params
    print 'EVALUATOR: running linear model with selected features'
    m = RegularizedRegression(config, {'batch_size': 8, 'num_train_steps': 1000}, intercept=False)
    dataset.set_active_split(config.train_suffix)
    m.train(dataset, '', features=features)
    dataset.set_active_split(config.test_suffix)
    print 'EVALUATOR: linear inference on test...'
    regression_predictions = m.inference(dataset, '')

    print 'EVALUATOR: running fixed model with selected features'
    m = FixedRegression(config, {'batch_size': 8, 'num_train_steps': 1000}, intercept=False)
    dataset.set_active_split(config.train_suffix)
    m.train(dataset, '', features=features)
    dataset.set_active_split(config.test_suffix)
    print 'EVALUATOR: fixed inference on test...'
    fixed_predictions = m.inference(dataset, '')



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
            regression_preds = regression_predictions.scores[var['name']]
            fixed_preds = fixed_predictions.scores[var['name']]
            assert len(regression_preds) == len(labels)
            assert len(fixed_preds) == len(labels)

            performance[var['name']] = eval_performance(
                var=var,
                labels=labels,
                regression_preds=regression_preds,
                fixed_preds=fixed_preds,
                dataset=dataset)

    mean_correlation = np.mean(correlations.values())
    mean_reg_performance = np.mean([d['regression']['stat'] for d in performance.values()])
    mean_fixed_performance = np.mean([d['fixed']['stat'] for d in performance.values()])
    return {'correlations': correlations, 
            'performance': performance, 
            'mu_corr': mean_correlation, 
            'mu_reg_perf': mean_reg_performance,
            'mu_fixed_perf': mean_fixed_performance}


