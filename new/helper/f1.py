from helper.ds import ds
import numpy as np

#calculates f1 score (both 2/3 and 1) for a set of data, given predicted affinities
def calc_f1(X, Y, times, preds, T, thres=1e-5):
    results = np.array([0.0, 0.0])
    for i in range(len(times)-1):
        start, stop = int(times[i]), int(times[i+1])
        num_people = int(np.sqrt(stop-start))+1

        pred = ds(preds[start:stop], num_people, thres) #gets predicted groupings
        truth = ds(Y[start:stop], num_people, thres) #gets actual groupings

        TF, FN, FP, P, R = indiv_f1(pred, truth, T) #calculates individual f1
        results += np.array([P, R]) #stores results

    results /= (len(times) - 1)
    P, R = results

    if P+R==0: f1 = 0 #edge case
    else: f1 = 2 * P * R / (P + R)

    return P, R, f1

# calculates true positives, false negatives, and false positives
# given the guesses, the true groups, and the threshold T
def indiv_f1(pred, truth, T):
    TP, FN, FP = 0, 0, 0
    n_true_groups = len(truth)
    n_pred_groups = len(pred)

    for true_group in truth:
        for pred_group in pred:
            n_found = 0
            for person in pred_group:
                if person in true_group:
                    n_found += 1

            n_total = max(len(true_group), len(pred_group))
            acc = float(n_found) / n_total
            if acc>=T: TP += 1

    #edge cases
    if n_true_groups == 0 and n_pred_groups == 0:
        return [0, 0, 0, 1, 1]
    elif n_true_groups == 0:
        return [0, n_pred_groups, 0, 0, 1]
    elif n_pred_groups == 0:
        return [0, 0, n_true_groups, 1, 0]
    else:
        FP = n_pred_groups - TP #false positive
        FN = n_true_groups - TP #false negative
        P = float(TP) / (TP + FP) #precision
        R = float(TP) / (TP + FN) #recall
        return TP, FN, FP, P, R
