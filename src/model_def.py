from keras.layers import Input, Dense, Conv1D, Embedding, concatenate, \
    GlobalMaxPooling1D, Dropout
from keras.models import Model
import numpy as np

# returns two tensors
# one for input_content, the other for tensor before final classification
def content2rep(vocab_size, max_len, drop_out=0.5,
                filter=200, dense_size=256, embed_dim=300):
    input_content = Input(shape=(max_len,),
                          name='Content_input')
    embed_layer = Embedding(vocab_size, embed_dim, input_length=max_len,
                            name='Embed')
    e_i = embed_layer(input_content)

    """TODO Confirm that there is no dropout/activation here"""
    # convolutional layers
    conv_out = []
    for kernel_size in range(1, 6):
        c = Conv1D(filter, kernel_size, activation='relu',
                   name='conv_' + str(kernel_size))(e_i)
        c = GlobalMaxPooling1D(name='max_pooling_' + str(kernel_size))(c)
        """TODO confirm that the dropout comes after max pooling"""
        c = Dropout(drop_out, name='drop_out_' + str(kernel_size))(c)
        conv_out.append(c)
    agg = concatenate(conv_out)

    dense_layer = Dense(dense_size, activation='relu')
    """TODO confirm that there is no dropout here"""
    content_rep = dense_layer(agg)

    return input_content, content_rep

class NN_architecture:

    def __init__(self,
                 vocab_size, max_len, drop_out=0.5,
                 filter=200, dense_size=256, embed_dim=300,
                 pretrained_weight_dir=None, weight_in_keras=None,
                 include_context=False, context_dense_size=None,
                 include_splex=False, splex_dense_size=None):
        # changeable hyper parameter
        self.vocab_size, self.max_len, self.drop_out = vocab_size, max_len, drop_out
        # hyper parameters that is mostly fixed
        self.filter, self.dense_size, self.embed_dim = filter, dense_size, embed_dim
        # pretrained_weight directory
        self.pretrained_weight_dir, self.weight_in_keras = pretrained_weight_dir, weight_in_keras
        # modifications to the model
        self.include_context, self.include_splex = include_context, include_splex
        self.input_content, self.content_rep = content2rep(self.vocab_size, self.max_len, self.drop_out,
                                                           self.filter, self.dense_size, self.embed_dim)
        inputs = [self.input_content]
        
        if pretrained_weight_dir is not None:
            self.load_pretrained_weights()

        concatenated_rep = self.content_rep
        
        """TODO implement the MLP for context and Splex"""
        if include_context:
            pass
        if include_splex:
            pass

        self.out_layer = Dense(3,
                               name='classification')
        out = self.out_layer(concatenated_rep)
        self.model = Model(inputs=inputs, outputs=out)
        self.model.summary()

    # loading the pretrained weight that from input to second last layer
    def load_pretrained_weights(self):
        if self.pretrained_weight_dir is not None and self.weight_in_keras:
            out = Dense(3)(self.content_rep)
            _model_ = Model(inputs=self.input_content, outputs=out)
            _model_.load_weights(self.pretrained_weight_dir)

if __name__ == '__main__':
    nn = NN_architecture(vocab_size=30000, max_len=150,
                         pretrained_weight_dir=None, weight_in_keras=True)
