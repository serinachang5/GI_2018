from keras.callbacks import EarlyStopping, ModelCheckpoint
from model_def import NN_architecture
from data_loader import Data_loader
from generator_util import create_clf_data, simplest_tweet2data
import numpy as np
import subprocess
from sklearn.metrics import precision_recall_fscore_support

def calculate_class_weight(y_train):
    class_count = np.sum(y_train, axis=0)
    class_count = 1 / class_count
    class_count /= np.sum(class_count)
    class_weight = dict([(idx, class_count[idx]) for idx in range(len(class_count))])
    return class_weight

class Experiment:

    def __init__(self, tweet2data, experiment_dir, comments='', epochs=20, patience=4, **kwargs):
        # creating the experiment dir
        # automatically generate a README
        if experiment_dir[:-1] != '/':
            experiment_dir += '/'
        experiment_dir = '../experiments/' + experiment_dir
        self.experiment_dir, self.kwargs = experiment_dir, kwargs
        subprocess.call(['rm', '-rf', experiment_dir])
        subprocess.call(['mkdir', experiment_dir])
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
            # retriving cross validataion data
            fold_data = self.dl.cv_data(fold_idx)
            ((X_train, y_train), (X_val, y_val), (X_test, y_test)) = \
                create_clf_data(self.tweet2data, fold_data, return_generators=False)
            class_weight = calculate_class_weight(y_train)

            # initializing model, train and predict
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
        np.savetxt('result_by_fold.np', results)
        np.savetxt('result_averaged.np', np.mean(results, axis=0))
        np.savetxt('result_std.np', np.std(results, axis=0))

        avg_macro_f = np.mean(np.mean(results, axis=0)[2])
        with open(self.experiment_dir + 'README', 'w') as readme:
            readme.write('macro F-score: %.4f\n' % avg_macro_f)

if __name__ == '__main__':
    options = ['word']
    experiment = Experiment(tweet2data=simplest_tweet2data, experiment_dir='test', options=options)
    experiment.cv()
