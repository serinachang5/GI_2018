from keras.layers import Input, Dense, Conv1D, Embedding, concatenate, \
    GlobalMaxPooling1D, Dropout
from keras.models import Model
import pandas as pd

def print_results_from_np(result_np):
    result_np = result_np.T
    d = [{'precision': r[0], 'recall': r[1], 'f-score': r[2], 'support': r[3]} for r in result_np]
    df = pd.DataFrame(d)
    print(df)

# returns two tensors
# one for input_content, the other for tensor before final classification
def content2rep(option, vocab_size, max_len, drop_out=0.5,
                filter=200, dense_size=256, embed_dim=300,
                kernel_range=(1,6)):
    input_content = Input(shape=(max_len,),
                          name= option + '_content_input')
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
                 word_vocab_size=30000, word_max_len=50,
                 char_vocab_size=1000, char_max_len=140,
                 drop_out=0.5,
                 filter=200, dense_size=256, embed_dim=300, kernel_range=range(1,6),
                 pretrained_weight_dir=None, weight_in_keras=None,
                 context_dim=None, context_dense_size=None,
                 splex_dense_size=None):
        """
        Initilizing a neural network architecture according to the specification

        Parameters
        ----------
        options: an array containing all the options considered in the neural network model ['char', 'word', 'context']
                    (probably splex in the future)
                    for each option, the input is mapped to a lower dimension,
                    then the lower dimension representation of each option is concatenated
                    and is followed by the final classification layer
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
        context_dim: the dimension of context representation
        context_dense_size: the dense layer size right before the context representation
        splex_dense_size: dense layer size right before the splex reps
        """
        self.options = options
        # changeable hyper parameter
        self.drop_out = drop_out
        self.word_vocab_size, self.word_max_len = word_vocab_size, word_max_len
        self.char_vocab_size, self.char_max_len = char_vocab_size, char_max_len
        # hyper parameters that is mostly fixed
        self.filter, self.dense_size, self.embed_dim, self.kernel_range = filter, dense_size, embed_dim, kernel_range
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

        if len(last_tensors) >= 2:
            concatenated_rep = concatenate(last_tensors)
        else:
            concatenated_rep = last_tensors[0]

        self.out_layer = Dense(3, activation='softmax',
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
