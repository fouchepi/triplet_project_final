from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools


def knn_classification_sklearn(trX_vect, trY_vect, teX_vect, teY_vect, n_nei):
    """
    Compute confusion matrix with knn classification predictions from sklearn
    """
    knn_class = KNeighborsClassifier(n_neighbors=n_nei)
    knn_class.fit(trX_vect, trY_vect)
    predictions = knn_class.predict(teX_vect)
    pred_proba = knn_class.predict_proba(teX_vect)
    conf_mat = confusion_matrix(teY_vect, predictions)
    return predictions, conf_mat


def knn_classification_top(trX_vect, trY_vect, teX_vect, teY_vect, n_nei=5):
    """
    Compute confusion matrix with knn classification predictions (classic and top n_nei) from sklearn
    """
    knn_class = KNeighborsClassifier(n_neighbors=n_nei)
    knn_class.fit(trX_vect, trY_vect)
    predictions = knn_class.predict(teX_vect)

    kneighbors = knn_class.kneighbors(teX_vect, return_distance=False)
    predictions_top = np.array([teY_vect[i] if teY_vect[i] in kn else predictions[i] for i, kn in enumerate(np.array(trY_vect)[kneighbors])])

    conf_mat = confusion_matrix(teY_vect, predictions)
    conf_mat_top = confusion_matrix(teY_vect, predictions_top)
    return predictions, predictions_top, conf_mat, conf_mat_top


def print_acc_classes(cm, classes):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i, c in enumerate(classes):
        sort = np.argsort(cm[i])[::-1]
        print("Accuracy of {} : {:.02f} %, max {} : {:.02f} %, 2nd {} : {:.02f} %"
              .format(c, cm[i][i] * 100, sort[0], cm[i][sort[0]] * 100, sort[1], cm[i][sort[1]] * 100))


def plot_confusion_matrix(cm, classes, normalize, size, save_name):
    """
    Plot the confusion matrix

    :param cm: confusion matrix to plot
    :param classes: name of the classes
    :param normalize: between 0 and 1
    :param save_name: name of the image
    """
    filename = '/data/data_pierre/conf_matrix/' + save_name + '.pdf'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        filename = '/data/data_pierre/conf_matrix/' + save_name + '_norm.pdf'
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=size)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix : ' + save_name)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)


def get_label(labels):
    """
    Return the predicted label.
    Handle ties by removing the farthest.

    :param labels: list of the best labels ordered by nearness
    :return: returns the predicted label.
    """
    while True:
        best = Counter(labels).most_common()
        if len(best) == 1:
            return best[0][0]
        if best[0][1] > best[1][1]:
            # return the most frequent label
            return best[0][0]
        else:
            # remove the farthest label
            labels = labels[:-1]