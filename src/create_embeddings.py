"""
===================
create_embeddings
===================
Author: Ethan Adams
Date: 04/21/2018
Simple script to train Word Embeddings on the indices that are in Data Loader.
"""


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
	option = 'word'
	max_len = 53
	vocab_size = 30000

	print('Initializing Data Loader')
	dl = Data_loader(vocab_size=vocab_size, max_len=max_len, option=option)

	# params for Word2Vec
	sentences = MySentences(dl)
	size = 300
	window = 5
	min_count = 5

	# create the model
	print('Training Word2Vec')
	model = gensim.models.Word2Vec(sentences, size=size, window=window, min_count=min_count)

	# save the model (as binary)
	out_file = 'index_word_embs_s{0}_w{1}_mc{2}.bin'.format(size, window, min_count)
	model.wv.save_word2vec_format(out_file, binary=True)
	print('Model saved to', out_file)




if __name__ == '__main__':
		main(sys.argv)