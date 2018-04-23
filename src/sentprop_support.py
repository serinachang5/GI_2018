"""
===================
sentprop_support
===================
Authors: Serina Chang
Date: 04/22/2018
Prepare embeddings for SENTPROP and evaluate resulting scores.
"""

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
    loss = [tok.lower() for tok in loss]
    print(loss)
    agg = [tok.lower() for tok in agg]
    print(agg)
    sub = [tok.lower() for tok in sub]
    print(sub)

    seed_file = 'seeds_' + specs + '_lower_p2.pkl'  # lowercased and protocol 2
    with open(seed_file, 'wb') as f:
        pickle.dump([loss, agg, sub], f, protocol=2)
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


def eval_seeds(splex):
    loss, agg, sub = pickle.load(open('seeds_hc_lower_p2.pkl', 'rb'))
    print('LOSS SEED SET')
    for word in loss:
        if word in splex:
            loss_i, agg_i, sub_i = splex[word]
            print(word, round(loss_i, 4), round(agg_i, 4), round(sub_i, 4))
    print('AGG SEED SET')
    for word in agg:
        if word in splex:
            loss_i, agg_i, sub_i = splex[word]
            print(word, round(loss_i, 4), round(agg_i, 4), round(sub_i, 4))
    print('SUBSTANCE SEED SET')
    for word in sub:
        if word in splex:
            loss_i, agg_i, sub_i = splex[word]
            print(word, round(loss_i, 4), round(agg_i, 4), round(sub_i, 4))


if __name__ == '__main__':
    # embs = pickle.load(open('svd_word_s300.pkl', 'rb'))
    # print('Loaded embeddings:', len(embs))
    # write_embeddings(embs, specs='svd_word_s300')
    #
    # loss, agg, sub = pickle.load(open('seeds_hc.pkl', 'rb'))
    # prep_seed_sets(loss, agg, sub, specs='hc')

    # raw_splex = pickle.load(open('splex_raw_svd_word_s300_seeds_hc.pkl', 'rb'), encoding='latin1')
    # print('Loaded raw SPLex:', len(raw_splex))
    # make_normalized(raw_splex, 'svd_word_s300_seeds_hc')

    normalized_splex = pickle.load(open('splex_normalized_svd_word_s300_seeds_hc.pkl', 'rb'))
    eval_seeds(normalized_splex)