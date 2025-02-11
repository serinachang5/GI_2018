import sys
from read_results import get_fold_macro_f
import numpy as np

if __name__ == '__main__':
    dir_name = sys.argv[1]
    header = [dir_name] + ['fold' + str(i + 1) for i in range(5)] + ['average']
    row = ['run' + str(i + 1) for i in range(5)] + ['average', 'max', 'min', 'diff']
    
    all_results = []
    for _ in range(5):
        fs = get_fold_macro_f(dir_name + '_' + str(_)).tolist()
        fs.append(np.mean(fs))
        all_results.append(fs)

    m = [np.mean([all_results[i][j] for i in range(5)]) for j in range(5)]
    m.append(np.mean(m))
    all_results.append(m)

    m = [np.max([all_results[i][j] for i in range(5)]) for j in range(5)]
    m.append(np.mean(m))
    all_results.append(m)

    m = [np.min([all_results[i][j] for i in range(5)]) for j in range(5)]
    m.append(np.mean(m))
    all_results.append(m)

    m = [(all_results[6][j] - all_results[7][j]) for j in range(5)]
    m.append(np.mean(m))
    all_results.append(m)

    all_results = np.around((np.array(all_results) * 100), decimals=1).tolist()
    print(', '.join(header))

    for r in range(9):
        print(', '.join([row[r]] + [str(x) for x in all_results[r]]))



