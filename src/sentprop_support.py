"""
===================
sentprop_support
===================
Authors: Serina Chang
Date: 04/22/2018
Prepare embeddings for SENTPROP and evaluate resulting scores.
"""

from sentence_tokenizer import int_array_rep, unicode_rep
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


'''SET-UP'''
# protocol 2 because SENTPROP is in Python 2
def write_embeddings(embs, specs):
	written_file = 'written_' + specs + '.txt'
	lex_file = 'lexicon_' + specs + '_p2.pkl'  # protocol 2
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
            idx = int_array_rep(word)[0]
            group_as_idx.append(str(idx))
        seeds.append(group_as_idx)

    seed_file = 'seeds_' + specs + '_idx_p2.pkl'  # as idx and protocol 2
    with open(seed_file, 'wb') as f:
        pickle.dump(seeds, f, protocol=2)
    print('Saved seed sets in', seed_file)

'''EVAL'''
def make_normalized(splex, specs):
    scores = np.array([x[1] for x in splex.items()])
    print('Scores:', scores.shape)  # expecting 20000 x 3

    # normalize scores across class to (0,1) range
    by_class = scores.T
    loss_scaler = MinMaxScaler()
    loss_scaler.fit(by_class[0].reshape(-1,1))
    agg_scaler = MinMaxScaler()
    agg_scaler.fit(by_class[1].reshape(-1,1))
    sub_scaler = MinMaxScaler()
    sub_scaler.fit(by_class[2].reshape(-1,1))

    normalized_splex = {}
    for word,(loss_raw, agg_raw, sub_raw) in splex.items():
        loss_scaled = loss_scaler.transform(loss_raw)[0][0]
        agg_scaled = agg_scaler.transform(agg_raw)[0][0]
        sub_scaled = sub_scaler.transform(sub_raw)[0][0]
        normalized_splex[word] = (loss_scaled, agg_scaled, sub_scaled)

    splex_file = 'splex_normalized_' + specs + '.pkl'
    with open(splex_file, 'wb') as f:
        pickle.dump(normalized_splex, f)
    print('Saved normalized SPLex in', splex_file)

def eval_seeds(loss, agg, sub, splex):
    print('LOSS SEED SET')
    for idx in loss:
        if idx in splex:
            loss_i, agg_i, sub_i = splex[idx]
            word = unicode_rep([int(idx)])
            print('{}, idx={}: loss={}, agg={}, sub={}'.format(
                word, idx, round(loss_i, 4), round(agg_i, 4), round(sub_i, 4)))
        else:
            print('Missing {}, idx={}'.format(unicode_rep([int(idx)]), idx))
    print('AGG SEED SET')
    for idx in agg:
        if idx in splex:
            loss_i, agg_i, sub_i = splex[idx]
            word = unicode_rep([int(idx)])
            print('{}, idx={}: loss={}, agg={}, sub={}'.format(
                word, idx, round(loss_i, 4), round(agg_i, 4), round(sub_i, 4)))
        else:
            print('Missing {}, idx={}'.format(unicode_rep([int(idx)]), idx))
    print('SUBSTANCE SEED SET')
    for idx in sub:
        if idx in splex:
            loss_i, agg_i, sub_i = splex[idx]
            word = unicode_rep([int(idx)])
            print('{}, idx={}: loss={}, agg={}, sub={}'.format(
                word, idx, round(loss_i, 4), round(agg_i, 4), round(sub_i, 4)))
        else:
            print('Missing {}, idx={}'.format(unicode_rep([int(idx)]), idx))

def eval_top_words(splex):
    for idx in range(1,101):
        loss_i, agg_i, sub_i = splex[str(idx)]
        word = unicode_rep([idx])
        print('{}, idx={}: loss={}, agg={}, sub={}'.format(
            word, idx, round(loss_i, 4), round(agg_i, 4), round(sub_i, 4)))

def sample_usage():
    test_indices = [str(idx) for idx in range(10)]

    raw_splex_file = 'splex_raw_svd_word_s300_seeds_hc.pkl'
    raw_splex = pickle.load(open(raw_splex_file, 'rb'), encoding='latin1')
    print('Number of embeddings in {}: {}'.format(raw_splex_file, len(raw_splex)))
    for idx in test_indices:
        if idx in raw_splex:
            print(unicode_rep([int(idx)]), idx, raw_splex[idx])
        else:
            print('No embedding found for', unicode_rep([int(idx)]),  idx)

    normed_splex_file = 'splex_normalized_svd_word_s300_seeds_hc.pkl'
    normed_splex = pickle.load(open(normed_splex_file, 'rb'))
    print('Number of embeddings in {}: {}'.format(normed_splex_file, len(normed_splex)))
    for idx in test_indices:
        if idx in normed_splex:
            print(unicode_rep([int(idx)]), idx, normed_splex[idx])
        else:
            print('No embedding found for', unicode_rep([int(idx)]),  idx)

if __name__ == '__main__':
    # embs = pickle.load(open('svd_word_s300.pkl', 'rb'))
    # print('Loaded embeddings:', len(embs))
    # write_embeddings(embs, specs='svd_word_s300')

    # loss, agg, sub = pickle.load(open('seeds_hc.pkl', 'rb'))
    # prep_seed_sets(loss, agg, sub, specs='hc')

    # raw_splex = pickle.load(open('splex_raw_svd_word_s300_seeds_hc.pkl', 'rb'), encoding='latin1')
    # print('Loaded raw SPLex:', len(raw_splex))
    # make_normalized(raw_splex, 'svd_word_s300_seeds_hc')

    # normalized_splex = pickle.load(open('splex_normalized_svd_word_s300_seeds_hc.pkl', 'rb'))
    # loss, agg, sub = pickle.load(open('seeds_hc_idx_p2.pkl', 'rb'))
    # eval_seeds(loss, agg, sub, normalized_splex)
    # eval_top_words(normalized_splex)

    sample_usage()