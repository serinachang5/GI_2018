from nn_experiment import Experiment
import pickle as pkl

def merge_dict(dicts, without_classification=True):
    result = {}
    for d in dicts:
        for key in d:
            if not (without_classification and 'classification' in key):
                result[key] = d[key]
    return result

if __name__ == '__main__':
    options = ['word']
    num_runs = 5
    ds_agg_weight = pkl.load(open('../weights/ds_word/pretrained_weights_as_numpy_0.p', 'rb'))
    ds_loss_weight = pkl.load(open('../weights/ds_word/pretrained_weights_as_numpy_1.p', 'rb'))
    pretrained_weight_dirs = merge_dict([ds_agg_weight, ds_loss_weight])
    pretrained_weight_dirs = [pretrained_weight_dirs] * 5

    for run_idx in range(num_runs):
        experiment = Experiment(options=options, pretrained_weight_dirs=pretrained_weight_dirs,
                                by_fold=True,
                                experiment_dir='ds_word_' + str(run_idx))
        experiment.cv()
