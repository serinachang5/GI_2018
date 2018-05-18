'''
Created on May 18, 2018

@author: siddharth
'''
from data_loader import Data_loader
import sys
import os
import numpy as np
import pickle
from model_def import NN_architecture
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


def save_as_numpy(model, save_dir, class_idx):

    layers = model.layers
    layers_dict = {}
    for layer in layers:
        layers_dict[layer.name] = layer.get_weights()

    print (layers_dict.keys())
    pickle.dump(layers_dict, open(os.path.join(save_dir, 'pretrained_weights_as_numpy_' + str(class_idx) + '.p'), "wb"))


def main(pretrained_weight_dirs = None, options = ['word'], patience = 7, save_dir = None, epochs = 100):

    dl = Data_loader(option = options[0], vocab_size = 40000)
    X_train, y_train, X_val, y_val = dl.distant_supv_data()
    # initialize the predictions
    num_val = y_val.shape[0]
    y_pred_val = [None] * num_val

    for class_idx in range(2):

        # create layer name that has prefix
        # since for each fodl we train model for aggression and loss models separately
        if class_idx == 0:
            prefix = 'aggression'
        else:
            prefix = 'loss'

        # initialize a model
        model = NN_architecture(options = options, prefix = prefix, pretrained_weight_dirs = pretrained_weight_dirs).model
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

        # create the label for this binary classification task

        _y_train_ = np.asarray([1 if v == class_idx else 0 for v in y_train])
        _y_val_ = np.asarray([1 if v == class_idx else 0 for v in y_val])

        # call backs
        es = EarlyStopping(patience = patience, monitor = 'val_loss', verbose = 1)
        weights_file = os.path.join(save_dir, 'pretrained_weights_' + str(class_idx) + '.weight')
        mc = ModelCheckpoint(weights_file, save_best_only = True, save_weights_only = True)
        callbacks = [es, mc]

        # training
        model.fit(x = X_train, y = _y_train_,
                       validation_data = (X_val, _y_val_))
        history = model.fit(x = X_train, y = _y_train_,
                                 validation_data = (X_val, _y_val_),
                                 callbacks = callbacks, epochs = epochs)

        # sometimes adam will stuck at a saddle point
        # if adam does not work, use adadelta
        # which will not get stuck but have lower performance
        losses = history.history['loss']
        if np.min(losses) > 0.2:
            print ('Using Adadelta optimizer!')
            model = NN_architecture(options = options, prefix = prefix, pretrained_weight_dirs = pretrained_weight_dirs).model
            model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
            model.fit(x = X_train, y = _y_train_,
                           validation_data = (X_val, _y_val_),
                           callbacks = callbacks, epochs = epochs)

        model.load_weights(weights_file)

        save_as_numpy(model, save_dir, class_idx)

        _y_pred_val_score = model.predict(X_val).flatten()

        np.savetxt(save_dir + 'class_%d_' % (class_idx) + 'pred_val.np', _y_pred_val_score)

        # threshold tuning
        best_t, best_f_val = 0, -1
        for t in np.arange(0.01, 1, 0.01):
            y_val_pred_ = [0] * num_val
            for idx in range(num_val):
                if y_pred_val[idx] is None and _y_pred_val_score[idx] >= t:
                    y_val_pred_[idx] = 1
            f = f1_score(_y_val_, y_val_pred_)
            if f > best_f_val:
                best_f_val = f
                best_t = t
            # a temp variable that we do not want its value
            # to be accidentally accessed by outside code
            y_val_pred_ = None

        # predictions made only when predictions not made by the previous model
        # and larger than the best threshold
        # true for both val_pred and test_pred
        for idx in range(num_val):
            if y_pred_val[idx] is None and _y_pred_val_score[idx] >= best_t:
                y_pred_val[idx] = class_idx

    # predict the rest as the "Other" class
    for idx in range(num_val):
        if y_pred_val[idx] is None:
            y_pred_val[idx] = 2

    np.savetxt(save_dir + 'pred_val.np', y_pred_val)
    np.savetxt(save_dir + 'truth_val.np', y_val)

    # append the result on this fold to results
    print(classification_report(y_val, y_pred_val, target_names = ['Aggression', 'Loss', 'Other'], digits = 4))


if __name__ == '__main__':

    # sys.argv[1] : directory where models should be saved
    # sys.argv[2] : path to pre-trained embeddings file

    # save_dir = '/home/siddharth/workspace/GI-DL/Experiments/emnlp_paper/runs/run10'
    pretrained_weight_dirs = ({'aggression_word_embed': [sys.argv[2]],
                               'loss_word_embed': [sys.argv[2]]})
    main(pretrained_weight_dirs, save_dir = sys.argv[1], epochs = 1)
