"""
===================
sentprop_support
===================
Authors: Serina Chang
Date: 04/26/2018
Prepare embeddings for SENTPROP algorithm. Post-process and evaluate resulting scores.
"""

from data_loader import Data_loader
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print('Initializing Data Loader')
dl = Data_loader()

'''SET-UP'''
# pickle in protocol 2 because SENTPROP is in Python 2
def write_embeddings(embs, specs):
	written_file = '../data/written_' + specs + '.txt'
	lex_file = '../data/lexicon_' + specs + '_p2.pkl'
	tokens = set()

	with open(written_file, 'w') as f:
		for word in embs:
			tokens.add(word)
			result = word + '\t'
			emb = embs[word]
			result += ' '.join([str(x) for x in emb])
			result += '\n'
			f.write(result)
	print('Wrote embeddings to', written_file)

	with open(lex_file, 'wb') as f:
		pickle.dump(tokens, f, protocol=2)
	print('Saved tokens in', lex_file)


def prep_seed_sets(loss, agg, sub, specs):
    seeds = []
    for group in [loss, agg, sub]:
        group_as_idx = []
        for word in group:
            idx = dl.convert2int_arr(word)[0]
            print(word, idx, dl.convert2unicode([idx]))
            group_as_idx.append(str(idx))
        seeds.append(group_as_idx)

    seed_file = '../data/seeds_' + specs + '_idx_p2.pkl'
    with open(seed_file, 'wb') as f:
        pickle.dump(seeds, f, protocol=2)
    print('Saved seed sets in', seed_file)


'''EVAL'''
def save_scaled(splex, specs, scale_type='minmax'):
    sorted_splex = sorted(splex.items(), key=lambda x:int(x[0]))
    indices = [x[0] for x in sorted_splex]
    scores = np.array([x[1] for x in sorted_splex])

    assert(scale_type == 'minmax' or scale_type == 'standard')
    if scale_type == 'minmax': # per feature, subtract min and divide by range -> (0,1) range
        scaled_scores = MinMaxScaler().fit_transform(scores)
        scaled_splex = dict((indices[i], scaled_scores[i]) for i in range(len(indices)))
        splex_file = '../data/splex_' + scale_type + '_' + specs + '.pkl'
        with open(splex_file, 'wb') as f:
            pickle.dump(scaled_splex, f)
        print('Saved scaled splex in', splex_file)
    else: # per feature, scale mean to 0 and variance to 1
        scaled_scores = StandardScaler().fit_transform(scores)
        scaled_splex = dict((indices[i], scaled_scores[i]) for i in range(len(indices)))
        splex_file = '../data/splex_' + scale_type + '_' + specs + '.pkl'
        with open(splex_file, 'wb') as f:
            pickle.dump(scaled_splex, f)
        print('Saved scaled splex in', splex_file)

# scale by .5 but reward loss vs aggression based on whichever one is stronger
def save_loss_agg_balanced(splex, specs):
    sorted_splex = sorted(splex.items(), key=lambda x:int(x[0]))
    indices = [x[0] for x in sorted_splex]
    scores = np.array([x[1] for x in sorted_splex])
    scaled_scores = np.zeros(scores.shape, dtype=np.float)
    for i in range(scores.shape[0]):
        orig_loss, orig_agg, orig_sub = scores[i]
        loss_agg_sum = orig_loss + orig_agg
        loss_scaled = orig_loss * (orig_loss/loss_agg_sum) if loss_agg_sum > .01 else orig_loss * .5
        agg_scaled = orig_agg * (orig_agg/loss_agg_sum) if loss_agg_sum > .01 else orig_agg * .5
        sub_scaled = orig_sub * .5
        scaled_scores[i] = [loss_scaled, agg_scaled, sub_scaled]
        if i > 100 and i < 105:
            print('Orig:', scores[i])
            print('New:', scaled_scores[i])

    scaled_splex = dict((indices[i], scaled_scores[i]) for i in range(len(indices)))
    splex_file = '../data/splex_balanced_' + specs + '.pkl'
    with open(splex_file, 'wb') as f:
        pickle.dump(scaled_splex, f)
    print('Saved balanced splex file to', splex_file)

def eval_seeds(loss, agg, sub, splex):
    print('LOSS SEED SET')
    for idx in loss:
        if idx in splex:
            loss_i, agg_i, sub_i = splex[idx]
            word = dl.convert2unicode([int(idx)])
            print('{}, idx={}: loss={}, agg={}, sub={}'.format(
                word, idx, round(loss_i, 4), round(agg_i, 4), round(sub_i, 4)))
        else:
            print('Missing {}, idx={}'.format(dl.convert2unicode([int(idx)]), idx))
    print('AGG SEED SET')
    for idx in agg:
        if idx in splex:
            loss_i, agg_i, sub_i = splex[idx]
            word = dl.convert2unicode([int(idx)])
            print('{}, idx={}: loss={}, agg={}, sub={}'.format(
                word, idx, round(loss_i, 4), round(agg_i, 4), round(sub_i, 4)))
        else:
            print('Missing {}, idx={}'.format(dl.convert2unicode([int(idx)]), idx))
    print('SUBSTANCE SEED SET')
    for idx in sub:
        if idx in splex:
            loss_i, agg_i, sub_i = splex[idx]
            word = dl.convert2unicode([int(idx)])
            print('{}, idx={}: loss={}, agg={}, sub={}'.format(
                word, idx, round(loss_i, 4), round(agg_i, 4), round(sub_i, 4)))
        else:
            print('Missing {}, idx={}'.format(dl.convert2unicode([int(idx)]), idx))

def eval_top_words(splex):
    print('TOP 100 WORDS')
    for idx in range(1,101):
        loss_i, agg_i, sub_i = splex[str(idx)]
        word = dl.convert2unicode([idx])
        print('{}, idx={}: loss={}, agg={}, sub={}'.format(
            word, idx, round(loss_i, 4), round(agg_i, 4), round(sub_i, 4)))

def sample_usage():
    test_indices = [str(idx) for idx in range(10)]

    raw_splex_file = 'splex_raw_svd_word_s300_seeds_hc.pkl'
    raw_splex = pickle.load(open(raw_splex_file, 'rb'), encoding='latin1')
    print('Number of embeddings in {}: {}'.format(raw_splex_file, len(raw_splex)))
    for idx in test_indices:
        if idx in raw_splex:
            print(dl.convert2unicode([int(idx)]), idx, raw_splex[idx])
        else:
            print('No embedding found for', dl.convert2unicode([int(idx)]),  idx)

    # same usage for minmax, standard, balanced_minmax, and balanced_standard
    scaled_splex_file = 'splex_minmax_svd_word_s300_seeds_hc.pkl'
    scaled_splex = pickle.load(open(scaled_splex_file, 'rb'))
    print('Number of embeddings in {}: {}'.format(scaled_splex_file, len(scaled_splex)))
    for idx in test_indices:
        if idx in scaled_splex:
            print(dl.convert2unicode([int(idx)]), idx, scaled_splex[idx])
        else:
            print('No embedding found for', dl.convert2unicode([int(idx)]),  idx)


if __name__ == '__main__':
    # PREP FOR SENTPROP
    # embs = pickle.load(open('../data/svd_word_s300.pkl', 'rb'))
    # print('Loaded embeddings:', len(embs))
    # write_embeddings(embs, specs='svd_word_s300')
    # loss, agg, sub = pickle.load(open('../data/seeds_hc.pkl', 'rb'))
    # prep_seed_sets(loss, agg, sub, specs='hc')

    # SCALING RAW OUTPUT
    # raw_splex = pickle.load(open('../data/splex_raw_svd_word_s300_seeds_hc.pkl', 'rb'), encoding='latin1')
    # print('Loaded raw SPLex:', len(raw_splex))
    # save_scaled(raw_splex, scale_type='minmax', specs='svd_word_s300_seeds_hc')
    # save_scaled(raw_splex, scale_type='standard', specs='svd_word_s300_seeds_hc')

    # LOSS-AGGRESSION BALANCING
    # splex = pickle.load(open('../data/splex_standard_svd_word_s300_seeds_hc.pkl', 'rb'))
    # save_loss_agg_balanced(splex, specs='standard_svd_word_s300_seeds_hc')
    # splex = pickle.load(open('../data/splex_minmax_svd_word_s300_seeds_hc.pkl', 'rb'))
    # save_loss_agg_balanced(splex, specs='minmax_svd_word_s300_seeds_hc')

    splex = pickle.load(open('../data/splex_standard_svd_word_s300_seeds_hc.pkl', 'rb'))
    loss, agg, sub = pickle.load(open('../data/seeds_hc_idx_p2.pkl', 'rb'))
    eval_seeds(loss, agg, sub, splex)
    eval_top_words(splex)

    # sample_usage()