import argparse
import sys
from sklearn.metrics import f1_score, accuracy_score
import math
import pandas as pd
from scipy.stats import bernoulli
import numpy as np


def load_labels(path, label_dict=None):
    with open(path, "r") as fp:
        if label_dict is None:
            labels = [label.strip() for label in fp]
            label_dict = {}
            for label in labels:
                if label not in label_dict:
                    label_dict[label] = len(label_dict)
            labels = np.array([label_dict[label] for label in labels])
            return labels, label_dict
        else:
            labels = np.array([label_dict[label.strip()] for label in fp])
            return labels

def compute_macro_f1(gold_labels, system_labels):
    return f1_score(gold_labels, system_labels)

def compute_accuracy(gold_labels, system_labels):
    return accuracy_score(gold_labels, system_labels)

def get_test_statistic(metric_name, gold_labels):
    if metric_name == "f1":
        metric = compute_macro_f1
    elif metric_name == "acc":
        metric = compute_accuracy
    else:
        raise Exception("Invalid metric name!")
    
    def test_statistic(sys1, sys2):
        obs1 = metric(gold_labels, sys1)
        obs2 = metric(gold_labels, sys2)
        return abs(obs1 - obs2)
    return test_statistic

def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    """
    b = np.random.random(a.shape)
    idx = np.argsort(b, axis=axis)
    shuffled = a[np.arange(a.shape[0])[:, None], idx]
    return shuffled

def test_significance(metric, gold_labels, system_labels1, system_labels2,
                      num_samples):
    test_statistic = get_test_statistic(metric, gold_labels)
    t = test_statistic(system_labels1, system_labels2)

    print("t = |{}(SYS1) - {}(SYS2)| = {:6.4f}".format(
        metric, metric, t))

    labels = np.vstack([system_labels1, system_labels2]).T
    print("Total labels = {}".format(labels.shape[0]))
    
    t_samples = []
    for i in range(num_samples):
        sample = scramble(labels)
        t_sample = test_statistic(sample[:,0], sample[:,1])
        t_samples.append(t_sample)

    results = pd.DataFrame(
        [[ts, ts > t] for ts in t_samples], columns=["t(X,Y)", "t(X,Y) > t"])

    r = results["t(X,Y) > t"].values.sum()

    pd.options.display.max_rows = 25
    print("")
    print("t={:8.6f}".format(t))
    print(results)
    print("")
    print("r={}".format(r))
    pval = (r + 1) / (num_samples + 1)
    print("p={:8.6f}".format(pval))
    return pval




def main():
    usage = """
    python statsig.py --gold-labels [GOLD]
                      --system-labels [SYS1 SYS2]
                      --metric [METRIC]
                      --samples [SAMPLES]
                      --class-conditional (optional)

    Computes statistical significance using the approximate randomization
    test. Label files should have one label per line.
    
    * GOLD should be a path to the ground truth labels.
    * SYS1, SYS2 should be the paths to the system produced labels.
    * METRIC is the performance measure you are testing for significance.
          Valid options are:
             f1  -- computes macro F1 score.
             acc -- computes accuracy.
    * SAMPLES is the number of samples to use. Above 1000 is better.
    * class-condtional -- when this flag is on, data is split into
                          separate groups based on ground truth label
                          and stat sig is computed indepently for 
                          each group.
    """

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--gold-labels", required=True, type=str)
    parser.add_argument("--system-labels", required=True, type=str, nargs=2)
    parser.add_argument(
        "--metric", choices=["f1", "acc"], default="f1")
    parser.add_argument("--samples", default=1000, type=int)
    parser.add_argument(
        "--class-conditional", action="store_true", default=False)
    parser.add_argument("--seed", default=9628635, type=int)
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    gold_labels, label_dict = load_labels(args.gold_labels)
    system_labels1 = load_labels(args.system_labels[0], label_dict=label_dict)
    system_labels2 = load_labels(args.system_labels[1], label_dict=label_dict)
    
    print(f1_score(gold_labels, system_labels1, average=None))
    print(f1_score(gold_labels, system_labels2, average=None))
    
    if len(gold_labels) != len(system_labels1):
        sys.stderr.write("Size of gold and system 1 labels does not agree!\n")
        sys.stderr.flush()
        sys.exit(1)

    if len(gold_labels) != len(system_labels2):
        sys.stderr.write("Size of gold and system 2 labels does not agree!\n")
        sys.stderr.flush()
        sys.exit(1)

    if args.class_conditional:
        pvals = []
        sorted_labels = sorted(label_dict.keys())
        for label in sorted_labels:
            print("Computing significance for ground truth = {}".format(label))
            label_idx = label_dict[label]
            '''
            I = gold_labels == label_idx
            gold_labels_cc = gold_labels[I]
            system_labels1_cc = system_labels1[I]
            system_labels2_cc = system_labels2[I]
            '''
            gold_labels_cc = gold_labels == label_idx
            system_labels1_cc = system_labels1 == label_idx
            system_labels2_cc = system_labels2 == label_idx
            
            pval = test_significance(
                args.metric, gold_labels_cc, system_labels1_cc, system_labels2_cc, 
                args.samples)
            print("")
            pvals.append(pval)
        
        print("Summary Results")
        print("===============")
        with open('.tmp.txt', 'w') as out_file:
            print(label_dict)
            for label, pval in zip(sorted_labels, pvals):
                print("{}: pval={:8.6f}".format(label, pval))
                out_file.write("{}: pval={:8.6f}\n".format(label, pval))
    

    else:
        test_significance(
            args.metric, gold_labels, system_labels1, system_labels2, args.samples)
  
if __name__ == "__main__":
    main()
