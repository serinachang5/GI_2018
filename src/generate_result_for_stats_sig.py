import sys
import numpy as np

dir_name = '../experiments/' + sys.argv[1]
num_runs = int(sys.argv[2])
num_folds = 5
num_class = 3

def make_int_array(np_arr):
    result = []
    for _ in np_arr:
        if _ > 1.5:
            result.append(2)
        elif _ > 0.5:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)

if __name__ == '__main__':

    # generating prediction
    all_pred = []
    for fold_idx in range(num_folds):
        voters = []
        for run_idx in range(num_runs):
            pred_test_fold = np.loadtxt('%s_%d/fold_%d_pred_test.np'
                                        % (dir_name, run_idx, fold_idx))
            pred_test_fold = make_int_array(pred_test_fold)
            num_test_fold = pred_test_fold.shape[0]
            one_hot_fold = np.zeros((num_test_fold, num_class))
            one_hot_fold[np.arange(num_test_fold), pred_test_fold] = 1
            voters.append(one_hot_fold)
        votes = np.sum(voters, axis=0)
        all_pred += np.argmax(votes, axis=-1).tolist()
    np.savetxt('../experiments/predictions/%s_pred.np' % sys.argv[1], all_pred)

    '''
    # generating gold label
    all_pred = []
    for fold_idx in range(num_folds):
        voters = []
        for run_idx in range(num_runs):
            pred_test_fold = np.loadtxt('%s_%d/fold_%d_truth_test.np'
                                        % (dir_name, run_idx, fold_idx))
            pred_test_fold = make_int_array(pred_test_fold)
            num_test_fold = pred_test_fold.shape[0]
            one_hot_fold = np.zeros((num_test_fold, num_class))
            one_hot_fold[np.arange(num_test_fold), pred_test_fold] = 1
            voters.append(one_hot_fold)
        votes = np.sum(voters, axis=0)
        all_pred += np.argmax(votes, axis=-1).tolist()
    np.savetxt('../experiments/predictions/gold.np', all_pred)
    '''


