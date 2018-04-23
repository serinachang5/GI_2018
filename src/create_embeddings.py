"""
===================
create_embeddings
===================
Authors: Ethan Adams & Serina Chang
Date: 04/22/2018
Train embeddings on the tweets in Data Loader (labeled + unlabeled data).
"""

import argparse
from data_loader import Data_loader
import gensim
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# Train on tweets in index form
def generate_w2v_embs(sentences, option):
	size = args['dim']
	min_count = args['min_count']
	window = args['window']
	iter = args['iter']

	print('Training Word2Vec...')
	model = gensim.models.Word2Vec(sentences, size=size, window=window,
								   min_count=min_count, iter=iter)
	print('Finished. Vocab size:', len(model.wv.vocab))

	# save the model (as binary)
	out_file = 'w2v_{0}_s{1}_w{2}_mc{3}_it{4}.bin'.format(option, size, window,
														  min_count, iter)
	model.wv.save_word2vec_format(out_file, binary=True)
	print('Word2Vec model saved to', out_file)


# Train on tweets in unicode form
def generate_svd_embs(sentences, option):
	size = args['dim']

	# get positive pointwise mutual info matrix
	mat, vocab = get_ppmi(sentences)

	# singular value decomposition - find most important eigenvalues
	u,s,v = np.linalg.svd(mat)
	print('Computed SVD')
	u = np.array(u)  # convert from matrixlib

	# make dictionary of unigram : embedding (truncated)
	embs = {}
	for i, word in enumerate(vocab):
		embs[word] = u[i, :size]
	print('Embedding dim:', len(list(embs.values())[0]))

	# save as pickle file
	out_file = 'svd_{0}_s{1}.pkl'.format(option, size)
	with open(out_file, 'wb') as f:
		pickle.dump(embs, f)
	print('SVD embeddings saved to', out_file)


def get_ppmi(sentences):
	count_model = CountVectorizer(lowercase=True, max_features=20000)
	counts = count_model.fit_transform(sentences)
	counts.data = np.fmin(np.ones(counts.data.shape), counts.data)  # want occurence, not count
	n,v = counts.shape  # n is num of docs, v is vocab size
	print('n = {}, v = {}'.format(n,v))
	vocab = sorted(count_model.vocabulary_.items(), key=lambda x: x[1])  # sort by idx
	vocab = [x[0] for x in vocab]
	print('First 10 words in vocab:', vocab[:10])

	coo = (counts.T).dot(counts)  # co-occurence matrix is v by v
	coo.setdiag(0)  # set same-word to 0
	coo = coo + np.full(coo.shape, .0001)  # smoothing

	marginalized = coo.sum(axis=0)  # smoothed num of coo per word
	prob_norm = coo.sum()  # smoothed sum of all coo
	print('Prob_norm:', prob_norm)
	row_mat = np.ones((v, v), dtype=np.float)
	for i in range(v):
		prob = marginalized[0,i] / prob_norm
		row_mat[i,:] = prob
	col_mat = row_mat.T
	joint = coo / prob_norm

	P = joint / (row_mat * col_mat)  # elementwise
	P = np.fmax(np.zeros((v, v), dtype=np.float), np.log(P))  # all elements >= 0
	print('Computed PPMI:', P.shape)
	return P, vocab


def main(args):
	# params for data loader
	max_len = 53
	vocab_size = 30000
	option = args['option']

	print('Initializing Data Loader')
	dl = Data_loader(vocab_size=vocab_size, max_len=max_len, option=option)

	mode = args['mode']
	assert(mode == 'w2v' or mode == 'svd')
	if mode == 'w2v':
		sentences = []
		for tweet in dl.all_data():
			# need in index form
			sentences.append([str(x) for x in tweet['int_arr']])
		generate_w2v_embs(sentences, option)
	else:
		sentences = []
		for tweet in dl.all_data():
			# need in unicode form
			sentences.append(dl.convert2unicode(tweet['int_arr']))
		print('Check sentence0:', sentences[0])
		generate_svd_embs(sentences, option)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('-opt', '--option', type = str, default = 'word', help = 'embedding option: {\'word\', \'char\'}')
	parser.add_argument('-md', '--mode', type = str, default = 'w2v', help = 'mode of embedding: {\'w2v\', \'svd\'}')

	parser.add_argument('-dim', '--dim', type = int, default = 300, help = 'dimension of embeddings')
	parser.add_argument('-min', '--min_count', type = int, default = 5, help = 'min_count for word2vec; ignored if svd')
	parser.add_argument('-win', '--window', type = int, default = 5, help = 'window for word2vec; ignored if svd')
	parser.add_argument('-it', '--iter', type = int, default = 20, help = 'iterations for word2vec; ignored if svd')

	args = vars(parser.parse_args())
	print(args)

	main(args)