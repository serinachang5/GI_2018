from keras.callbacks import EarlyStopping, ModelCheckpoint
from model_def import NN_architecture
from data_loader import Data_loader
from generator_util import create_clf_data, simplest_tweet2data
import numpy as np
import subprocess
from sklearn.metrics import precision_recall_fscore_support
from keras import backend as K

# calculating the classweight given the y_train
# the weight will be inversely proprotional to the label
# belonging to that class
def calculate_class_weight(y_train):
    class_count = np.sum(y_train, axis=0)
    class_count = 1 / class_count
    class_count /= np.sum(class_count)
    class_weight = dict([(idx, class_count[idx]) for idx in range(len(class_count))])
    return class_weight

# for supervised learning without any pre-training
# we need to adjust the vocabulary size for all the content input
# mapping every word that occurs less than twice in the training set to 1
def adapt_vocab(X_train, X_list):
    for key in X_train:
        if key in [option + '_content_input' for option in ['char', 'word']]:
            if 'char' in key:
                threshold = 3
            else:
                threshold = 2
            wc = {}
            for xs in X_train[key]:
                for x in xs:
                    if wc.get(x) is None:
                        wc[x] = 0
                    wc[x] += 1
            def f(x):
                return x if (wc.get(x) is not None and wc[x] >= threshold) else 1
            X_train[key] = np.array([[f(x) for x in xs] for xs in X_train[key]])
            for X in X_list:
                X[key] = np.array([[f(x) for x in xs] for xs in X[key]])

# an experiment class that runs cross validation
class Experiment:

    def __init__(self, tweet2data, experiment_dir, adapt_train_vocab=False,
                 comments='', epochs=20, patience=4, **kwargs):
        """
        an experiment class that runs cross validation
        designed to enable easy experiments with combinations of:
        1) context representation:
            handled by tweet2data, implement the value for key "context_input"
            include "context" in options
        2) pre-training methods:
            handled by pretrained_weight_dir in the kwargs argument
            None if there is no pretraining weight available
        3) char vs. word:
            specified in "options"
            options = ['char', 'word'] if you want to include both
            implement the value for key "word_content_input"
        options = ['char', 'word', 'context'] if you want to include everything

        Parameters
        ----------
        tweet2data: a function that maps a tweet dictionary to a dictionary that contains (key, val) pairs s.t.
                    key is recognizable by the neural network model,
                    val is a numpy array
                    see example in generator_util.py, simplest_tweet2data
        experiment_dir: the directory that the experiment weights and results will be saved
        adapt_train_vocab: under supervised training without pretraining,
                            some vocab will not be seen (twice) in the training set.
                            if set to True, then vocab occuring less than twice will be removed.
        comments: the comments that will be written to the README
        epochs: number of epochs of training during cross validation
        patience: number of epochs allowable for not having any improvement on the validation set
        kwargs: arguments that will be passed to initializing the neural network model (shown below)

        ========== below is the parameters needed by the neural network model ==========

        options: an array containing all the options considered in the neural network model ['char', 'word', 'context']
                    (probably splex in the future)
                    for each option, the input is mapped to a lower dimension,
                    then the lower dimension representation of each option is concatenated
                    and is followed by the final classification layer
        word_vocab_size: number of word level vocabs to be considered
        word_max_len: number of words in a tweet sentence
        char_vocab_size: number of char level vocabs to be considered
        char_max_len: number of chars in a tweet sentence
        drop_out: dropout rate for regularization
        filter: number of filters for each kernel size
        dense_size: the size of the dense layer following the max pooling layer
        embed_dim: embedding dimension for character and word level
        kernel_range: range of kernel sizes
        pretrained_weight_dir: a dictionary containing the pretrained weight.
                    e.g. {'char': '../weights/char_ds.weights'} means that the pretrained weight for character level model
                    is in ../weights/char_ds.weights
        weight_in_keras: whether the weight is in Keras
        context_dim: the dimension of context representation
        context_dense_size: the dense layer size right before the context representation
        splex_dense_size: dense layer size right before the splex reps
        """
        # creating the experiment dir
        # automatically generate a README
        if experiment_dir[:-1] != '/':
            experiment_dir += '/'
        experiment_dir = '../experiments/' + experiment_dir
        self.experiment_dir, self.kwargs = experiment_dir, kwargs
        subprocess.call(['rm', '-rf', experiment_dir])
        subprocess.call(['mkdir', experiment_dir])
        self.adapt_train_vocab = adapt_train_vocab
        with open(self.experiment_dir + 'README', 'w') as readme:
            readme.write(comments + '\n')
            for key in kwargs:
                readme.write("%s: %s\n" % (str(key), str(kwargs[key])))

        # initializing fields of the class
        self.tweet2data = tweet2data
        self.fold = 5
        self.dl = Data_loader(option='both', labeled_only=True, **kwargs)
        self.epochs, self.patience = epochs, patience

    def cv(self):
        results = []

        for fold_idx in range(self.fold):
            print('cross validation fold %d.' % fold_idx)

            # retriving cross validataion data
            fold_data = self.dl.cv_data(fold_idx)
            ((X_train, y_train), (X_val, y_val), (X_test, y_test)) = \
                create_clf_data(self.tweet2data, fold_data, return_generators=False)
            if self.adapt_train_vocab:
                adapt_vocab(X_train, (X_val, X_test))

            class_weight = calculate_class_weight(y_train)

            # initializing model, train and predict
            K.clear_session()
            self.model = NN_architecture(**self.kwargs).model
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')

            # call backs
            es = EarlyStopping(patience=self.patience, verbose=1, mode='auto')
            weight_dir = self.experiment_dir + str(fold_idx) + '.weight'
            mc = ModelCheckpoint(weight_dir,
                                 save_best_only=True, save_weights_only=True)
            callbacks = [es, mc]

            # training
            self.model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), callbacks=callbacks,
                           epochs=self.epochs, class_weight=class_weight)
            self.model.load_weights(weight_dir)

            # prediction
            y_pred = self.model.predict(x=X_test)
            y_pred = np.argmax(y_pred, axis=-1)
            y_test = np.argmax(y_test, axis=-1)
            results.append(precision_recall_fscore_support(y_pred, y_test))

        # saving results
        results = np.array(results)
        np.savetxt(self.experiment_dir + 'result_by_fold.np', results.flatten())
        np.savetxt(self.experiment_dir + 'result_averaged.np', np.mean(results, axis=0))
        np.savetxt(self.experiment_dir + 'result_std.np', np.std(results, axis=0))

        avg_macro_f = np.mean(np.mean(results, axis=0)[2])
        with open(self.experiment_dir + 'README', 'a') as readme:
            readme.write('macro F-score: %.4f\n' % avg_macro_f)

if __name__ == '__main__':
    options = ['word']
    experiment = Experiment(tweet2data=simplest_tweet2data, experiment_dir='test', adapt_train_vocab=True,
                            options=options)
    experiment.cv()
