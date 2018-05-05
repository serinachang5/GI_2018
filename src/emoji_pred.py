from model_def import content2rep
from data_loader import Data_loader
import pickle as pkl
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random
import numpy as np


num_emoji = 64

def get_generator_step(X, y, batch_size=64, shuffle=False):
    return g(X, y, batch_size, shuffle), len(X) / batch_size

def g(X, y, batch_size=64, shuffle=False):
    num_data, idx = len(X), 0
    while True:
        if idx > num_data - batch_size:
            idx = (idx - batch_size) % num_data
        if idx % num_data == 0 and shuffle:
            order = [i for i in range(num_data)]
            random.shuffle(order)
            X, y = [X[i] for i in order], [y[i] for i in order]
        X_batch = np.array(X[idx:idx + batch_size])
        y_batch = np.zeros((batch_size, num_emoji))
        y_batch[np.arange(batch_size), y[idx:idx + batch_size]] = 1
        idx += batch_size
        yield X_batch, y_batch

special_tokens = set([0, 1, 2, 3])

def return_x_y(int_arr, emoji2idx, emoji_set):
    ts = set(int_arr)
    if 3 in ts:
        return int_arr, []
    flag = False
    for i in ts:
        if i not in special_tokens:
            flag = True
    if not flag:
        return int_arr, []

    dl.convert2unicode(int_arr)
    ys = ts & emoji_set
    x = [i for i in int_arr if i not in ys]
    while len(x) < len(int_arr):
        x.append(0)
    return x, [emoji2idx[y] for y in ys]

def get_x_y_from_list(tweet_dicts, emoji2idx, emoji_set, count):
    label2data = dict([(i, []) for i in range(num_emoji)])
    for tweet_dict in tweet_dicts:
        x, ys = return_x_y(tweet_dict['padded_int_arr'], emoji2idx, emoji_set)
        for y in ys:
            label2data[y].append(x)
    for y in label2data:
        random.shuffle(label2data[y])
        if len(label2data[y]) < count:
            label2data[y] = label2data[y] * int(count / len(label2data[y]) + 1)
        label2data[y] = label2data[y][:count]
    X, Y = [], []
    for y in label2data:
        X += label2data[y]
        Y += [y] * count
    random.shuffle(X), random.shuffle(Y)
    return X, Y

if __name__ == '__main__':
    debug = False

    # loading word.pkl and extract top emojis
    words = pkl.load(open('../model/word.pkl', 'rb'))
    idx2emojis = sorted([word for word in words if words[word]['isemoji']],
                        key=lambda emoji: -words[emoji]['occurence_in_unlabeled'])[:num_emoji]
    idx2emojis = [words[word]['id'] for word in idx2emojis]
    emoji2idx = dict([(idx2emojis[idx], idx) for idx in range(num_emoji)])
    emoji_set = set(idx2emojis)

    # loading data
    dl = Data_loader()
    if debug:
        print('Emojis being considered are ...')
        print(dl.convert2unicode(idx2emojis))
    tr, val = dl.unlabeled_tr_val()

    # getting training data
    X_train, y_train = get_x_y_from_list(tr, emoji2idx, emoji_set, count=24000)
    train_generator, train_step = get_generator_step(X_train, y_train, shuffle=True)

    if debug:
        for _ in range(10):
            print(dl.convert2unicode(X_train[_]))
            print(dl.convert2unicode([idx2emojis[y_train[_]]]))

    X_val, y_val = get_x_y_from_list(val, emoji2idx, emoji_set, count=6000)
    val_generator, val_step = get_generator_step(X_val, y_val, shuffle=False)

    # initializing model
    input, rep = content2rep()
    out = Dense(num_emoji, activation='softmax')(rep)
    model = Model(inputs=input, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    es = EarlyStopping(patience=4, monitor='val_accuracy', verbose=1, mode='max')
    weight_dir = 'pretrain_weights/emoji_pred.weight'
    mc = ModelCheckpoint(weight_dir,
                         save_best_only=True, save_weights_only=True)
    callbacks = [es, mc]

    # training
    model.fit_generator(train_generator, train_step, epochs=10, callbacks=callbacks,
                        validation_data=val_generator, validation_steps=val_step)


















