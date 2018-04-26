import pickle as pkl

user_dictionary = pkl.load(open('user.pkl', 'rb'))
anonymized_dictionary = dict([(user_dictionary[name]['id'], user_dictionary[name])
                              for name in user_dictionary])
pkl.dump(anonymized_dictionary, open('anonymized_user.pkl', 'wb'))
