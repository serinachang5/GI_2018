import numpy as np
import math


def get_config(config_type = 'ACL'):

    if config_type == 'ACL':
        top_loss_emojis = [u'\U0001f64f', u'\U0001f614', u'\U0001f513', u'\U0001f494', u'\U0001f613', u'\uffab', u'\U0001f62a', u'\U0001f193', u'\U0001f61e', u'\U0001f47c']
        top_agg_emojis = [u'\U0001f52b', u'\U0001f608', u'\U0001f637', u'\U0001f529', u'\uffab', u'\U0001f482', u'\U0001f4a5', u'\U0001f3b4', u'\U0001f635', u'\U0001f448']
        desired_dist = {'Loss': 14.87, 'Aggression': 6.67, 'Other': 78.46}
    elif config_type == 'EMNLP':
        top_loss_emojis = [u'\U0001f64f', u'\U0001f614', u'\U0001f494', u'\U0001f513', u'\U0001f613', u'\U0001f47c', u'\uffab', u'\U0001f61e', u'\U0001f62a', u'\U0001f622']
        top_agg_emojis = [u'\U0001f52b', u'\U0001f608', u'\U0001f637', u'\U0001f4a5', u'\U0001f691', u'\U0001f44f', u'\U0001f44e', u'\U0001f529', u'\U0001f635', u'\U0001f479']
        desired_dist = {'Loss': 11.26, 'Aggression': 5.14, 'Other': 83.6}
    else:
        return None

    return {'top_loss_emojis': top_loss_emojis, 'top_agg_emojis':top_agg_emojis, 'desired_dist':desired_dist}


def subsample(data, keep_num, seed = 45345):
    # just keep keep_num number of tweets
    shuffle_indices = np.random.RandomState(seed = seed).permutation(np.arange(len(data)))
    # shuffle_indices = np.random.permutation(np.arange(len(data)))
    shuffled_data = data[shuffle_indices]
    return shuffled_data[:keep_num]


def get_subsample_counts(actual_counts, desired_dist):
    _X_loss = (actual_counts['Loss'] * 100.0) / desired_dist['Loss']
    _X_agg = (actual_counts['Aggression'] * 100.0) / desired_dist['Aggression']
    _X_other = (actual_counts['Other'] * 100.0) / desired_dist['Other']
    _X_min = min(min(_X_agg, _X_loss), _X_other)
    print ('minimum X: ', _X_min)
    keep_loss = int(math.floor((_X_min * desired_dist['Loss']) / 100.0))
    keep_aggression = int(math.floor((_X_min * desired_dist['Aggression']) / 100.0))
    keep_other = int(math.floor((_X_min * desired_dist['Other']) / 100.0))
    print ('keep_loss: ', keep_loss)
    print ('keep_aggression: ', keep_aggression)
    print ('keep_other: ', keep_other)
    return {'Loss': keep_loss, 'Aggression': keep_aggression, 'Other':keep_other}
