import numpy as np

def helper_generator(X, y, batch_size, sample):
    counter = 0
    num_data = len(X)
    while True:
        if sample:
            idxes = np.randint(num_data, shape=(batch_size,))
        else:
            idxes = [idx % num_data for idx in range(counter, counter + batch_size)]
            counter += batch_size
        return dict([(key, X[key][idxes]) for key in X]), y[idxes]

def create_data(tweet2data, tweet_dicts, return_generators=False,
                batch_size=32, sample=False):
    """
    A map that takes in tweet dictionaries and return data points readable for keras fit/fit_generator

    Parameters
    ----------
    tweet2data: a function that maps each tweet dictionary to a data point that is recognizable by keras
                attributes include input_content, input_context, input_splex, y
    tweet_dicts: a list of tweets dictionary
    return_generators: whether (generator, step_size) is returned or (X, y) is returned

    Returns
    -------
    X: key-worded inputs
    y: one-hot labels
        OR
    generator: a generator that will generate
    step_size: number of times for a generator to complete one epoch

    """
    data = []
    keys = [key for key in tweet2data(tweet_dicts[0])]

    # convert each tweet_dict to a dictionary that only contains field that is recognizable and useulf
    # for the model
    for tweet_dict in tweet_dicts:
        data.append(tweet2data(tweet_dict))

    X = dict([(key, np.array([d[key] for d in data])) for key in keys])
    y = np.array([d['y'] for d in data])

    # return the entire datapoints and labels in one single array
    if not return_generators:
        return X, y

    generator = helper_generator(X, y, batch_size, sample)
    step_size = len(X) / batch_size
    return generator, step_size

def create_clf_data(tweet2data, tr_test_val_dicts, return_generators=False, batch_size=32):
    tr, val, test = tr_test_val_dicts
    return (create_data(tweet2data, tr, return_generators=return_generators, batch_size=batch_size, sample=True),
            create_data(tweet2data, val, return_generators=return_generators, batch_size=batch_size, sample=False),
            create_data(tweet2data, test, return_generators=return_generators, batch_size=batch_size, sample=False))

# an example of tweet2data
# takes in a tweet dictionary
# returns a dictionary that has key - value
# where keys are arguments recognizable by the model (matches input layer name)
# and values are corresponding numpy arrays
def simplest_tweet2data(tweet_dict):
    result = {}
    one_hot_labels = np.eye(3)
    if tweet_dict['label'] == 'Aggression':
        result['y'] = one_hot_labels[0]
    elif tweet_dict['label'] == 'Loss':
        result['y'] = one_hot_labels[1]
    else:
        result['y'] = one_hot_labels[2]
    result['content_input'] = tweet_dict['padded_int_arr']
    return result

if __name__ == '__main__':
    from data_loader import Data_loader
    option = 'word'
    max_len = 20
    vocab_size = 30000
    dl = Data_loader(vocab_size=vocab_size, max_len=max_len, option=option)
    fold_idx = 0
    data_fold = dl.cv_data(fold_idx)
    tr, val, test = data_fold
    print(tr[0])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_clf_data(simplest_tweet2data,
                                                                           data_fold)
    for key in X_train:
        print(X_train[key])
