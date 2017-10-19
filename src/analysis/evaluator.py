import numpy as np




def evaluate(config, dataset, predictions, model_dir):
    # predictions = abstract_model.Prediction
    performance = {}
    correlations = {}

    for var in config.data_spec[1:]:
        if var['control']:
            features = dataset.features
            input_text = dataset.get_tokenized_input()
            # TODO -- INSTEAD OF LOGITS, 
            #.        GATHER LABELS
            print predictions.scores
            print var['name']
            logits = predictions.scores[var['name']]
            print logits
            print np.argmax(logits, axis=0)
            quit()
            correlations[var] = np.mean(
                cramers_v(
                    feature=f, 
                    input=input_text, 
                    num_levels=dataset.num_levels(var['name']),
                    labels=None))
        else:
            pass




