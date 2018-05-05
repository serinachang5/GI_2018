"""
===================
model_def
===================
Author: Ruiqi Zhong
Date: 05/04/2018
This module includes a model class s.t. each component is exactly the same as the previous ACL paper
nevertheless, it allows combination of different models (concatenated at the last layer)
"""


from keras.layers import Input, Dense, Conv1D, Embedding, concatenate, \
    GlobalMaxPooling1D, Dropout
from keras.models import Model

# returns two tensors
# one for input_content, the other for tensor before final classification
def content2rep(option='word', vocab_size=40000, max_len=50, drop_out=0.5,
                filter=200, dense_size=256, embed_dim=300,
                kernel_range=(1,6)):

    # input layer
    input_content = Input(shape=(max_len,),
                          name= option + '_content_input')

    # embedding layer
    embed_layer = Embedding(vocab_size, embed_dim, input_length=max_len,
                            name= option + '_embed')
    e_i = embed_layer(input_content)
    embed_drop_out = Dropout(drop_out, name=option + '_embed_dropout')
    e_i = embed_drop_out(e_i)

    # convolutional layers
    conv_out = []
    for kernel_size in kernel_range:
        c = Conv1D(filter, kernel_size, activation='relu',
                   name= option + '_conv_' + str(kernel_size))(e_i)
        c = GlobalMaxPooling1D(name= option + '_max_pooling_' + str(kernel_size))(c)
        c = Dropout(drop_out, name= option + '_drop_out_' + str(kernel_size))(c)
        conv_out.append(c)
    agg = concatenate(conv_out)

    dense_layer = Dense(dense_size, activation='relu',
                        name= option + '_last')
    content_rep = dense_layer(agg)

    return input_content, content_rep


class NN_architecture:

    def __init__(self,
                 options,
                 input_dim_map=None,
                 word_vocab_size=40000, word_max_len=50,
                 char_vocab_size=1200, char_max_len=150,
                 drop_out=0.5,
                 filter=200, dense_size=256, embed_dim=300, kernel_range=range(1,6),
                 pretrained_weight_dir=None, weight_in_keras=None):
        """
        Initilizing a neural network architecture according to the specification
        access the actual model by self.model

        Parameters
        ----------
        options: an array containing all the options considered in the neural network model ['char', 'word']
                    (probably splex in the future)
                    for each option, the input is mapped to a lower dimension,
                    then the lower dimension representation of each option is concatenated
                    and is followed by the final classification layer
        input_dim_map: a map from additional input name to its dimension
        word_vocab_size: number of word level vocabs to be considered
        word_max_len: number of words in a tweet sentence
        char_vocab_size: number of char level vocabs to be considered
        char_max_len: number of chars in a tweet sentence
        drop_out: dropout rate for regularization
        filter: number of filters for each kernel size
        dense_size: the size of the dense layer following the max pooling layer
        embed_dim: embedding dimension for character and word level
        kernel_range: range of kernel sizes
        pretrained_weight_dir: a dictionary containing the pretrained weight.
                    e.g. {'char': '../weights/char_ds.weights'} means that the pretrained weight for character level model
                    is in ../weights/char_ds.weights
        weight_in_keras: whether the weight is in Keras
        """
        self.options = options
        if input_dim_map is None:
            input_dim_map = {}
        self.input_dim_map = input_dim_map

        # changeable hyper parameter
        self.drop_out = drop_out
        self.word_vocab_size, self.word_max_len = word_vocab_size, word_max_len
        self.char_vocab_size, self.char_max_len = char_vocab_size, char_max_len

        # hyper parameters that is mostly fixed
        self.filter, self.dense_size, self.embed_dim, self.kernel_range = filter, dense_size, embed_dim, kernel_range

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
        for option in self.options:

            # how to map char input to the last layer
            if option in ['char', 'word']:
                if option == 'char':
                    input_content, content_rep = content2rep(option,
                                                             self.char_vocab_size, self.char_max_len, self.drop_out,
                                                             self.filter, self.dense_size, self.embed_dim, self.kernel_range)
                else:
                    input_content, content_rep = content2rep(option,
                                                             self.word_vocab_size, self.word_max_len, self.drop_out,
                                                             self.filter, self.dense_size, self.embed_dim, self.kernel_range)
                inputs.append(input_content)
                last_tensors.append(content_rep)
                """TODO: Implement load weights here"""

        # directly concatenate addtional inputs (such as splex scores and context representations)
        # to the last layer
        for input_name in self.input_dim_map:
            input = Input(shape=(self.input_dim_map[input_name],),
                                      name=input_name + '_input')
            inputs.append(input)
            last_tensors.append(input)

        # concatenate all the representations
        if len(last_tensors) >= 2:
            concatenated_rep = concatenate(last_tensors)
        else:
            concatenated_rep = last_tensors[0]

        # out layer
        self.out_layer = Dense(3, activation='softmax',
                               name='classification')
        out = self.out_layer(concatenated_rep)
        self.model = Model(inputs=inputs, outputs=out)
        self.model.summary()

if __name__ == '__main__':
    options = ['char', 'word']
    nn = NN_architecture(options,
                         word_vocab_size=40000, word_max_len=50,
                         char_vocab_size=1200, char_max_len=150,
                         context_dim=300, context_dense_size=256,
                         pretrained_weight_dir=None, weight_in_keras=None)
