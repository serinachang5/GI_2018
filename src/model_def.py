from keras.layers import Input, Dense, Conv1D, Embedding, concatenate, \
    GlobalMaxPooling1D, Dropout
from keras.models import Model
import numpy as np

# returns two tensors
# one for input_content, the other for tensor before final classification
def content2rep(option, vocab_size, max_len, drop_out=0.5,
                filter=200, dense_size=256, embed_dim=300):
    input_content = Input(shape=(max_len,),
                          name= option + '_content_input')
    embed_layer = Embedding(vocab_size, embed_dim, input_length=max_len,
                            name= option + '_embed')
    e_i = embed_layer(input_content)

    """TODO Confirm that there is no dropout/activation here"""
    # convolutional layers
    conv_out = []
    for kernel_size in range(1, 6):
        c = Conv1D(filter, kernel_size, activation='relu',
                   name= option + '_conv_' + str(kernel_size))(e_i)
        c = GlobalMaxPooling1D(name= option + '_max_pooling_' + str(kernel_size))(c)
        """TODO confirm that the dropout comes after max pooling"""
        c = Dropout(drop_out, name= option + '_drop_out_' + str(kernel_size))(c)
        conv_out.append(c)
    agg = concatenate(conv_out)

    dense_layer = Dense(dense_size, activation='relu',
                        name= option + '_last')
    """TODO confirm that there is no dropout here"""
    content_rep = dense_layer(agg)

    return input_content, content_rep

class NN_architecture:

    def __init__(self,
                 options,
                 word_vocab_size, word_max_len,
                 char_vocab_size, char_max_len,
                 drop_out=0.5,
                 filter=200, dense_size=256, embed_dim=300,
                 pretrained_weight_dir=None, weight_in_keras=None,
                 context_dim=None, context_dense_size=None,
                 splex_dense_size=None):
        # changeable hyper parameter
        self.drop_out = drop_out
        self.word_vocab_size, self.word_max_len = word_vocab_size, word_max_len
        self.char_vocab_size, self.char_max_len = char_vocab_size, char_max_len
        # hyper parameters that is mostly fixed
        self.filter, self.dense_size, self.embed_dim = filter, dense_size, embed_dim
        # context dimension
        self.context_dim, self.context_dense_size = context_dim, context_dense_size
        self.splex_dense_size = splex_dense_size
        # pretrained_weight directory
        self.pretrained_weight_dirs, self.weight_in_keras = pretrained_weight_dir, weight_in_keras
        if self.pretrained_weight_dirs is None:
            self.pretrained_weight_dirs = {}
        if self.weight_in_keras is None:
            self.weight_in_keras = {}
        self.create_model()

    def create_model(self):
        # for each option, create computational graph and load weights
        inputs, last_tensors = [], []
        for option in options:

            # how to map char input to the last layer
            if option in ['char', 'word']:
                if option == 'char':
                    input_content, content_rep = content2rep(option,
                                                             self.char_vocab_size, self.char_max_len, self.drop_out,
                                                             self.filter, self.dense_size, self.embed_dim)
                else:
                    input_content, content_rep = content2rep(option,
                                                             self.word_vocab_size, self.word_max_len, self.drop_out,
                                                             self.filter, self.dense_size, self.embed_dim)
                inputs.append(input_content)
                last_tensors.append(content_rep)
                self.load_pretrained_weights(input_content, content_rep,
                                             self.pretrained_weight_dirs.get(option), self.weight_in_keras.get(option))

            # how to map context input to the last layer
            elif option == 'context':
                input_context = Input(shape=(self.context_dim,),
                                      name='context_input')
                context_layer = Dense(self.context_dense_size, activation='relu',
                                      name='context_last')
                context_rep = context_layer(input_context)
                inputs.append(input_context)
                last_tensors.append(context_rep)
            else:
                print('Option %s has not been implemented yet.' % option)

        concatenated_rep = concatenate(last_tensors)


        self.out_layer = Dense(3,
                               name='classification')
        out = self.out_layer(concatenated_rep)
        self.model = Model(inputs=inputs, outputs=out)
        self.model.summary()

    # loading the pretrained weight that from input to second last layer
    def load_pretrained_weights(self, input_content, content_rep, weight_dir, weight_in_keras):
        if weight_dir is not None and weight_in_keras:
            out = Dense(3)(content_rep)
            _model_ = Model(inputs=input_content, outputs=out)
            _model_.load_weights(self.pretrained_weight_dir)

if __name__ == '__main__':
    options = ['char', 'word', 'context']
    nn = NN_architecture(options,
                         word_vocab_size=30000, word_max_len=50,
                         char_vocab_size=1000, char_max_len=150,
                         context_dim=300, context_dense_size=256,
                         pretrained_weight_dir=None, weight_in_keras=None)
