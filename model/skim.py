import pickle as pkl
pkl_names = [s + '.pkl' for s in ['word', 'char', 'user']]
for pkl_name in pkl_names:
    name2property = pkl.load(open(pkl_name, 'rb'))
    id2property = [(name2property[name]['id'], (name, name2property[name])) for name in name2property]
    for _ in range(100):
        print(id2property[_][1])
        print('---------')
