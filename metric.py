import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import pdb

def precision_k(label_test, label_predict, k, remove_zero = False):
    # if remove_zero:
    #     label_filter = label_test != 0
    #     label_test = label_test[label_filter]
    #     label_predict = label_predict[label_filter]

    precision_k_neg = []
    precision_k_pos = []
    for i in range(len(label_test)):
        
        label_test_i = label_test[i]
        label_predict_i = label_predict[i]
        if remove_zero:
            label_filter = label_test_i != 0
            label_test_i = label_test_i[label_filter]
            label_predict_i = label_predict_i[label_filter]

        num_pos = 100
        num_neg = 100
        label_test_i = np.argsort(label_test_i)
        label_predict_i = np.argsort(label_predict_i)
        
        neg_test_set = set(label_test_i[:num_neg])
        pos_test_set = set(label_test_i[-num_pos:])
        neg_predict_set = set(label_predict_i[:k])
        pos_predict_set = set(label_predict_i[-k:])
        precision_k_neg.append(len(neg_test_set.intersection(neg_predict_set)) / k)
        precision_k_pos.append(len(pos_test_set.intersection(pos_predict_set)) / k)
        return np.mean(precision_k_neg), np.mean(precision_k_pos)

# def rmse(label_test, label_predict, remove_zero = False):
#     if remove_zero:
#         mask = label_test != 0
#         return np.sqrt(mean_squared_error(label_test, label_predict * mask))
#     else:
        return np.sqrt(mean_squared_error(label_test, label_predict))

def rmse(label_test, label_predict, remove_zero = False):
    if remove_zero:
        mask = label_test != 0
        return np.sqrt(mean_squared_error(label_test[mask], label_predict[mask]))
    else:
        return np.sqrt(mean_squared_error(label_test, label_predict))

def correlation(label_test, label_predict, correlation_type, remove_zero = False):
    if correlation_type == 'pearson':
        corr = pearsonr
    elif correlation_type == 'spearman':
        corr = spearmanr
    else:
        raise ValueError("Unknown correlation type: %s" % correlation_type)
    score = []
    if len(label_test.shape) == 1:
        if remove_zero:
            zero_filter = (label_test == 0)
            label_test = label_test[~zero_filter]
            label_predict = label_predict[~zero_filter]
        score.append(corr(label_test, label_predict)[0])
    else:
        for lb_test, lb_predict in zip(label_test, label_predict):
            if remove_zero:
                zero_filter = (lb_test == 0)
                lb_test = lb_test[~zero_filter]
                lb_predict = lb_predict[~zero_filter]
                if sum(lb_predict) == 0:
                    lb_predict[0] = 0.00001
            score.append(corr(lb_test, lb_predict)[0])
    if np.isnan(np.mean(score)):
        pdb.set_trace()
    return np.mean(score), score

