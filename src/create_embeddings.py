"""
===================
create_embeddings
===================
Author: Ethan Adams & Serina Chang
Date: 04/21/2018
Simple script to train Word Embeddings on the indices that are in Data Loader.
"""

import argparse
import gensim
from data_loader import Data_loader
import sys

# Class to yield sentences of our data
class MySentences(object):
    def __init__(self, data_loader):
        self.all_data = data_loader.all_data()
 
    def __iter__(self):
        for tweet in self.all_data:
                yield [str(x) for x in tweet['int_arr']] # word2vec expects strings

def main(args):
	# params for data loader
	max_len = 53
	vocab_size = 30000
	option = args['option']

	print('Initializing Data Loader')
	dl = Data_loader(vocab_size=vocab_size, max_len=max_len, option=option)

	# params for Word2Vec
	sentences = MySentences(dl)
	size = args['dim']
	min_count = args['min_count']
	window = args['window']
	iter = args['iter']

	# create the model
	print('Training Word2Vec')
	model = gensim.models.Word2Vec(sentences, size=size, window=window,
								   min_count=min_count, iter=iter)
	# save the model (as binary)
	out_file = 'index_word_embs_s{0}_w{1}_mc{2}_it{3}.bin'.format(size, window, min_count, iter)
	model.wv.save_word2vec_format(out_file, binary=True)
	print('Model saved to', out_file)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-opt', '--option', type = str, default = 'word', help = 'embedding option: {\'word\', \'char\'}')
    parser.add_argument('-md', '--mode', type = str, default = 'w2v', help = 'mode of embedding: {\'w2v\', \'svd\'}')

    parser.add_argument('-dim', '--dim', type = int, default = 300, help = 'dimension of embeddings')
    parser.add_argument('-min', '--min_count', type = int, default = 5, help = 'min_count for word2vec; ignored if svd')
    parser.add_argument('-win', '--window', type = int, default = 5, help = 'window for word2vec; ignored if svd')
    parser.add_argument('-it', '--iter', type = int, default = 20, help = 'iterations for word2vec; ignored if svd')

    args = vars(parser.parse_args())

    main(args)