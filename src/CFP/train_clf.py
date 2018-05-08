import argparse
import datetime
from numpy.random import choice
import os
import shutil
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import time

from Nadam import NadamOptimizer
from cnn_lm_nce import CNN_Model
import numpy as np
import tensorflow as tf
from train_cnn_lm_nce import get_dense_embeddings
from collections import namedtuple

from data_loader import Data_loader

def batch_iter(data, batch_size, shuffle = False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]

def dev_step(sess, model, x_batch, y_batch):

    """
    Evaluates model on a dev set based on the mode
    """
    feed_dict = {
      model.input_x: x_batch,
      model.input_y_clf: y_batch,
      model.dropout_keep_prob: 1.0
      # cnn.is_train: 0
    }
    loss, probabilities = sess.run(
        [model.clf_loss, model.probabilities],
        feed_dict)
    return loss, probabilities

def train_clf(sess, model, args, corpus):

    # Define Training procedure
    global_step = tf.Variable(0, name = "global_step", trainable = False)
    # optimizer = tf.train.AdamOptimizer(learning_rate = 1e-2)
    optimizer = NadamOptimizer()
    grads_and_vars = optimizer.compute_gradients(model.clf_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    clf_dir = 'clf_runs_' + str(args['fold_idx'])
    if os.path.isdir(os.path.join(args['model_save_dir'], clf_dir)):
        shutil.rmtree(os.path.join(args['model_save_dir'], clf_dir))
    out_dir = os.path.abspath(os.path.join(args['model_save_dir'], clf_dir))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", model.clf_loss)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
#             dev_summary_op = tf.summary.merge([loss_summary])
#             dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
#             dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep = 1)

    print('Initializing all variables . . .')
    sess.run(tf.global_variables_initializer())

    if args['pretrained_weights'] is not None:
        print('Restoring weights from existing checkpoint . . .')
        saver.restore(sess, args['pretrained_weights'])

    def train_step(x_batch, y_batch):
        """
        A single training step based on the mode
        """
        feed_dict = {
          model.input_x: x_batch,
          model.input_y_clf: y_batch,
          model.dropout_keep_prob: args['dropout']
          # cnn.is_train: 1
        }
        _, step, summaries, loss, probabilities = sess.run(
            [train_op, global_step, train_summary_op, model.clf_loss, model.probabilities],
            feed_dict)
        train_summary_writer.add_summary(summaries, step)

        return loss, probabilities

    X_tr = corpus.X_tr
    X_val = corpus.X_val
    X_test = corpus.X_test
    y_tr = corpus.y_tr
    y_val = corpus.y_val
    y_test = corpus.y_test

    best_val_f = 0
    best_val_probabilities = None
    patience = 5

    for epoch in range(args['n_epochs']):
        _tr_loss = 0
        # tr_probabilities = []
        print('Epoch %d:' % (epoch))
        _start = time.time()
        batches = batch_iter(
            list(zip(X_tr, y_tr)), args['batch_size'], shuffle = True)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            _loss, _ = train_step(x_batch, y_batch)
            _tr_loss += _loss
            # tr_probabilities.extend(probabilities)

        current_step = tf.train.global_step(sess, global_step)
        tr_loss = _tr_loss / len(X_tr)
        print('Run time: %d s' % (time.time() - _start))
        print('Training Loss: %f' % (tr_loss))

        val_batches = batch_iter(list(zip(X_val, y_val)), args['batch_size'])
        _val_loss = 0
        val_probabilities = []
        for batch in val_batches:
            x_batch, y_batch = zip(*batch)
            _loss, probabilities = dev_step(sess, model, x_batch, y_batch)
            _val_loss += _loss
            val_probabilities.extend(probabilities)

        val_loss = _val_loss / len(X_val)
        val_f = f1_score(np.argmax([[1, 0, 0] if x == 0 else [0, 0, 1] if x == 2 else [0, 1, 0] for x in y_val], axis = 1), np.argmax(val_probabilities, axis = 1), average = 'macro')

        print('Val Loss: %f, Val macro f: %f' % (val_loss, val_f))

        if val_f > best_val_f:
            best_val_f = val_f
            best_val_probabilities = val_probabilities
            path = saver.save(sess, checkpoint_prefix, global_step = current_step)
            print("Saved model checkpoint to {}\n".format(path))
            patience = 5
        else:
            patience -= 1
            print("\n")

        if patience == 0:
            print('Early stopping . . .')
            break

    print(classification_report(y_val, np.argmax(best_val_probabilities, axis = 1), target_names = corpus.class_names))

def main(args):

    dl = Data_loader()
    label2idx = dl.get_label2idx()
    idx2label = {v:k for k, v in label2idx.items()}
    class_weights_dict = dl.get_class_weights()
    class_weights = [class_weights_dict[idx2label[i]] for i in range(len(label2idx))]
    counts = dl.get_freq_counts()
    W = get_dense_embeddings(dl.token2id, args['emb_dim'], emb_file = args['emb_file'])

    #X_unld_tr, X_unld_val = dl.unlabeled_tr_val_as_arrays(min_len = 5, padded = True)

    #corpus = TweetCorpus(train_file = args['train_file'], val_file = args['val_file'], test_file = args['test_file'] , dictionaries_file = args['dictionaries_file'])

    Corpus = namedtuple('Corpus', 'label2idx idx2label token2idx idx2token counts class_weights W X_tr y_tr X_val y_val X_test y_test pad_token_idx class_names')

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args['ts'] = ts

    # print_hyper_params(args)

    pw = None

    if args['pretrained_weights'] is not None:
        pw = args['pretrained_weights']

    for fold_idx in range(5):
        X_tr, y_tr, X_val, y_val, X_test, y_test = dl.cv_data_as_arrays(fold_idx=fold_idx, min_len = 5, padded = True)

        y_tr = np.array([label2idx[y] for y in y_tr])
        y_val = np.array([label2idx[y] for y in y_val])
        y_test = np.array([label2idx[y] for y in y_test])

        #y_tr = np.array([[1, 0, 0] if 'loss' in x.lower() else [0, 0, 1] if 'aggr' in x.lower() else [0, 1, 0] for x in y_tr])
        #y_val = np.array([[1, 0, 0] if 'loss' in x.lower() else [0, 0, 1] if 'aggr' in x.lower() else [0, 1, 0] for x in y_val])
        #y_test = np.array([[1, 0, 0] if 'loss' in x.lower() else [0, 0, 1] if 'aggr' in x.lower() else [0, 1, 0] for x in y_test])

        corpus = Corpus(label2idx = label2idx,
                        idx2label = idx2label,
                        token2idx = dl.token2id,
                        idx2token = dl.id2token,
                        counts = counts,
                        class_weights = class_weights,
                        W = W,
                        X_tr = X_tr,
                        y_tr = y_tr,
                        X_val = X_val,
                        y_val = y_val,
                        X_test = X_test,
                        y_test = y_test,
                        pad_token_idx = 0,
                        class_names = [idx2label[0], idx2label[1], idx2label[2]]
                        )

        if pw is not None:
            suffix = 'model-29892'
            prefix = os.path.join(pw, 'lm_runs_' + str(fold_idx), 'checkpoints')

            for f in os.listdir(prefix):
                if f.startswith('model-'):
                    suffix = f.split('.')[0]
                    break
            
            args['pretrained_weights'] = os.path.join(prefix, suffix)

        args['fold_idx'] = fold_idx
        print ('Running fold: %d' % (fold_idx))
        
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
                
                print('List of trainable variables:')
                for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    print(i.name)

                # sess, model, args, corpus
                train_clf(sess, cnn, args, corpus)

def parse_arguments():

    parser = argparse.ArgumentParser(description = '')
    # even though short flags can be used in the command line, they can not be used to access the value of the arguments
    # i.e args['pt'] will give KeyError.
    parser.add_argument('-sdir', '--model_save_dir', type = str, help = 'directory where trained model should be saved')
    parser.add_argument('-w', '--pretrained_weights', type = str, default = None, help = 'Path to pretrained weights directory')
    parser.add_argument('-ef', '--emb_file', type = str, default = None, help = '.bin file containing pre-trained embeddings')
    parser.add_argument('-ed', '--emb-dim', type = int, default = 300, help = 'If emb-file is given then the value passed here will be ignored.')
    parser.add_argument('-epochs', '--n_epochs', type = int, default = 30)
    parser.add_argument('-nfmaps', '--nfeature_maps', type = int, default = 200)
    parser.add_argument('-do', '--dropout', type = float, default = 0.5)
    parser.add_argument('-bsz', '--batch_size', type = int, default = 256)

    args = vars(parser.parse_args())

    return args

if __name__ == '__main__':

    main(parse_arguments())
