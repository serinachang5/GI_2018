import argparse
import datetime
from sklearn.metrics import classification_report
from cnn_lm_nce import CNN_Model
import numpy as np
import tensorflow as tf
from train_clf import batch_iter
from train_clf import dev_step
import os

from data_loader import Data_loader
from collections import namedtuple
from train_cnn_lm_nce import get_dense_embeddings

def test_clf(sess, model, args, corpus):
    X_test = corpus.X_test
    y_test = corpus.y_test

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep = 1)

    print('Initializing all variables . . .')
    sess.run(tf.global_variables_initializer())

    if args['pretrained_weights'] is not None:
        print('Restoring weights from existing checkpoint . . .')
        saver.restore(sess, args['pretrained_weights'])

        test_batches = batch_iter(list(zip(X_test, y_test)), args['batch_size'])
        test_probabilities = []
        for batch in test_batches:
            x_batch, y_batch = zip(*batch)
            _, probabilities = dev_step(sess, model, x_batch, y_batch)
            test_probabilities.extend(probabilities)

    print(classification_report(y_test, np.argmax(test_probabilities, axis = 1), target_names = corpus.class_names, digits=4))

def main(args):

    #corpus = TweetCorpus(test_file = args['test_file'], dictionaries_file = args['dictionaries_file'])

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args['ts'] = ts

    #class_weights = [corpus.class_weights[i] for i in range(len(corpus.label2idx))]

    dl = Data_loader()
    label2idx = dl.get_label2idx()
    idx2label = {v:k for k, v in label2idx.items()}
    class_weights_dict = dl.get_class_weights()
    class_weights = [class_weights_dict[idx2label[i]] for i in range(len(label2idx))]
    counts = dl.get_freq_counts()
    W = get_dense_embeddings(dl.token2id, args['emb_dim'], emb_file = args['emb_file'])

    #X_unld_tr, X_unld_val = dl.unlabeled_tr_val_as_arrays(min_len = 5, padded = True)

    #corpus = TweetCorpus(train_file = args['train_file'], val_file = args['val_file'], test_file = args['test_file'] , dictionaries_file = args['dictionaries_file'])

    Corpus = namedtuple('Corpus', 'label2idx idx2label token2idx idx2token counts class_weights W X_test y_test pad_token_idx class_names')

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args['ts'] = ts

    # print_hyper_params(args)

    pw = None

    if args['pretrained_weights'] is not None:
        pw = args['pretrained_weights']

    # print_hyper_params(args)

    for fold_idx in range(5):
        _, _, _, _, X_test, y_test = dl.cv_data_as_arrays(fold_idx=fold_idx, min_len=5, padded=True)

        y_test = np.array([label2idx[y] for y in y_test])

        corpus = Corpus(label2idx = label2idx,
                        idx2label = idx2label,
                        token2idx = dl.token2id,
                        idx2token = dl.id2token,
                        counts = counts,
                        class_weights = class_weights,
                        W = W,
                        X_test = X_test,
                        y_test = y_test,
                        pad_token_idx = 0,
                        class_names = [idx2label[0], idx2label[1], idx2label[2]]
                        )

        if pw is not None:
            suffix = 'model-29892'
            prefix = os.path.join(pw, 'clf_runs_' + str(fold_idx), 'checkpoints')

            for f in os.listdir(prefix):
                if f.startswith('model-'):
                    suffix = f.split('.')[0]
                    break
            
            args['pretrained_weights'] = os.path.join(prefix, suffix)
        
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement = True,
                log_device_placement = False)
            sess = tf.Session(config = session_conf)
            with sess.as_default():
                cnn = CNN_Model(sequence_length = dl.max_len,
                                num_classes = len(label2idx),
                                vocab_size = dl.vocab_size,
                                embedding_size = len(corpus.W[0]),
                                filter_sizes = [1, 2, 3, 4, 5],
                                num_filters = args['nfeature_maps'],
                                embeddings = corpus.W,
                                class_weights = class_weights)
                
                # sess, model, args, corpus
                test_clf(sess, cnn, args, corpus)

def parse_arguments():

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-ef', '--emb_file', type = str, default = None, help = '.bin file containing pre-trained embeddings')
    parser.add_argument('-ed', '--emb-dim', type = int, default = 300, help = 'If emb-file is given then the value passed here will be ignored.')
    parser.add_argument('-w', '--pretrained_weights', type = str, default = None, help = 'Path to pretrained weights file')
    parser.add_argument('-nfmaps', '--nfeature_maps', type = int, default = 200)
    parser.add_argument('-bsz', '--batch_size', type = int, default = 256)

    args = vars(parser.parse_args())

    return args

if __name__ == '__main__':

    main(parse_arguments())
