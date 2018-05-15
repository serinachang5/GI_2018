from nn_experiment import Experiment
import pickle
from model_def import input_name_is_user_idx


# randomly initialized
def baseline():
    options = ['word']
    experiment = Experiment(experiment_dir='baseline',
                            pretrained_weight_dirs=None,
                            input_name2id2np=None,
                            options=options)
    experiment.cv()

# use list of modes to specify how to integrate word2vec and splex:
# 'wl' word-level - word embedding
# 'tl' tweet-level - built by TweetLevel, concatenated after CNN
# 'cl' context-level - built by Contextifier, concatenated after CNN
def nn_with_additions(w2v_modes, splex_modes, user_modes, interaction_dim=-1, run_num = None, testing=False):
    if len(w2v_modes) == 0 and splex_modes == 0:
        baseline()
        return

    dir_name = ''
    if len(w2v_modes) > 0:
        dir_name += 'W2V_' + '_'.join(w2v_modes)
    if len(splex_modes) > 0:
        dir_name += 'SP_' + '_'.join(splex_modes)
    if len(user_modes) > 0:
        dir_name += 'US_' + '_'.join(user_modes)
    if interaction_dim > 0:
        dir_name += '_' + str(interaction_dim)
    if run_num is not None:
        dir_name += '_' + str(run_num)
    print('Will save experiment in', dir_name)

    # include as word embedding in pretrained
    if 'wl' in w2v_modes and 'wl' in splex_modes:
        word_embs = 'word_emb_w2v_splex.np'
        word_embed_dim = 302
    elif 'wl' in w2v_modes:
        word_embs = 'word_emb_w2v.np'
        word_embed_dim = 300
    elif 'wl' in splex_modes:
        word_embs = 'word_emb_splex.np'
        word_embed_dim = 2
    else:
        word_embs = None
        word_embed_dim = None

    pretrained = {}
    if word_embs is not None:
        pretrained['aggression_word_embed'] = [word_embs]
        pretrained['loss_word_embed'] = [word_embs]

    # include as concatenated feature
    input_name2id2np = {}
    if 'tl' in splex_modes or 'cl' in w2v_modes or 'cl' in splex_modes or len(user_modes) > 0:
        all_inputs = pickle.load(open('input_name2id2np.pkl', 'rb'))
        if 'tl' in splex_modes:
            input_name2id2np['splex_tl'] = all_inputs['splex_tl']
        if 'cl' in w2v_modes:
            input_name2id2np['w2v_cl'] = all_inputs['w2v_cl']
        if 'cl' in splex_modes:
            input_name2id2np['splex_cl'] = all_inputs['splex_cl']
        if 'post' in user_modes:
            input_name2id2np['post_user_index'] = all_inputs['user_post']
        if 'rt' in user_modes:
            input_name2id2np['rt_user_index'] = all_inputs['user_retweet']

    check_input_name2id2np(input_name2id2np)
    options = ['word']

    if testing:
        experiment = Experiment(experiment_dir=dir_name,
                                pretrained_weight_dirs=pretrained,
                                input_name2id2np=input_name2id2np,
                                epochs=1, patience=1,
                                options=options,
                                embed_dim=word_embed_dim,
                                interaction_layer_dim=interaction_dim)
    else:
        experiment = Experiment(experiment_dir=dir_name,
                                pretrained_weight_dirs=pretrained,
                                input_name2id2np=input_name2id2np,
                                patience=7,
                                options=options,
                                embed_dim=word_embed_dim,
                                interaction_layer_dim=interaction_dim)
    experiment.cv()


def check_input_name2id2np(inputs):
    for input_name in inputs:
        if input_name_is_user_idx(input_name):
            print(input_name)
            id2np = inputs[input_name]
            for id in id2np:
                assert('int' in str(type(id2np[id])))

if __name__ == '__main__':
    nn_with_additions(w2v_modes=['wl'], splex_modes=['tl'], user_modes=['post'], run_num=1)