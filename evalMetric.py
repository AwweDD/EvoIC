from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np


def align_cluster_labels(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    label_map = dict(zip(col_ind, row_ind))
    return np.array([label_map[p] for p in y_pred if p in label_map] or [label_map.get(p, -1) for p in y_pred])


def cluster_metric(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    # pred_adjusted = align_cluster_labels(label, pred)
    acc = metrics.accuracy_score(label, pred)
    print("[Clustering Result]: ACC = {:.2f}, NMI = {:.2f}, ARI = {:.2f}".format(acc * 100, nmi * 100, ari * 100))
    return [acc, nmi, ari]
