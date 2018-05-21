import sys
import subprocess
import numpy as np

if __name__ == '__main__':

    dir_name = sys.argv[1]
    num_runs = int(sys.argv[2])
    ensemble_dir = '../experiments/' + str(dir_name) + '_ensemble'
    subprocess.call(['rm', '-rf', ensemble_dir])
    subprocess.call(['mkdir', ensemble_dir])
    experiment_dir = '../experiments/' + dir_name
    for name in ['val', 'test', 'ensemble']:
        for fold_idx in range(5):
            all_agg_scores = []
            all_loss_scores = []
            for run_idx in range(num_runs):
                agg_score= np.loadtxt(experiment_dir + '_'
                                      + str(run_idx) + '/fold_%d_class_0_pred_%s.np' % (fold_idx, name))
                all_agg_scores.append(agg_score)
                loss_score= np.loadtxt(experiment_dir + '_'
                                       + str(run_idx) + '/fold_%d_class_1_pred_%s.np' % (fold_idx, name))
                all_loss_scores.append(loss_score)
            fold_agg_scores, fold_loss_scores = (np.mean(all_agg_scores, axis=0),
                                                 np.mean(all_loss_scores, axis=0))
            
            with open(ensemble_dir + '/fold_%d_%s_pred.tsv' % (fold_idx, name), 'w') as out_file:
                out_file.write('Loss\tAggression\n')
                for idx in range(len(fold_agg_scores)):
                    out_file.write('%f\t%f\n' % (fold_loss_scores[idx], fold_agg_scores[idx]))

