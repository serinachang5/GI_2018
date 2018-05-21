import sys
import numpy as np
from read_results import np2df

if __name__ == '__main__':
    dir_name = sys.argv[1]
    num_runs = int(sys.argv[2])
    all_run_results = []
    for run_idx in range(num_runs):
        np_dir_name = '../experiments/' + dir_name + '_' + str(run_idx) + '/result_by_fold.np'
        result_by_fold = np.reshape(np.loadtxt(np_dir_name), (-1, 4, 3))
        run_results = np.mean(result_by_fold, axis=0)
        all_run_results.append(run_results)
    avg_run_results = np.mean(all_run_results, axis=0)
    print(np2df(avg_run_results))
    print('macro-f: %.3f' % np.mean(avg_run_results[2]))
