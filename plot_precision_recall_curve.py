from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import average_precision_score, confusion_matrix


def plot_stats(y_test, y_score, average_precision, figure_name):
    tn, fp, fn, tp = confusion_matrix(y_test, y_score).ravel()
    print((tn, fp, fn, tp))
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.clf()
    plt.figure(figsize=(8.0,5.0))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(figure_name)
    plt.show()
