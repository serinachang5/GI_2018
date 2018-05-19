from nn_experiment import Experiment

if __name__ == '__main__':
    options = ['word']
    for run_idx in range(5):
        experiment = Experiment(experiment_dir='word_' + str(run_idx),
                                input_name2id2np=None, adapt_train_vocab=True,
                                options=options)
        experiment.cv()

    options = ['char']
    for run_idx in range(5):
        experiment = Experiment(experiment_dir='char_' + str(run_idx),
                                input_name2id2np=None, adapt_train_vocab=True,
                                kernel_range=range(1, 6),
                                options=options)
        experiment.cv()

    options = ['word', 'char']
    for run_idx in range(5):
        experiment = Experiment(experiment_dir='charpword_' + str(run_idx),
                                input_name2id2np=None, adapt_train_vocab=True,
                                kernel_range=range(1, 6),
                                options=options)
        experiment.cv()
