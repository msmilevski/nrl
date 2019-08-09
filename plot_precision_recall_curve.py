from sklearn.metrics import precision_recall_curve
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from inspect import signature
import pandas as pd
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

def plot_2d_points(x, y, fig_name):
    colors = ['red', 'blue']
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(colors))

    cb = plt.colorbar()
    loc = np.arange(0, max(y), max(y) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(colors)
    plt.title("Data for baseline model")
    plt.xlabel("Jaccard similarity between sentences")
    plt.ylabel("Euclidian distance between image embeddings")
    plt.savefig(fig_name)

def plot_loss(file_name):
    experiment = file_name.split('/')[-3]
    data = pd.read_csv(file_name)

    train_loss = data['train_loss'].values
    val_loss = data['val_loss'].values

    train_aps = data['train_aps'].values
    val_aps = data['val_aps'].values
    fig = plt.figure(figsize=(8, 8))
    epochs = range(0, len(train_loss))
    plt.plot(epochs, train_loss)
    plt.plot(epochs, val_loss)
    plt.title(experiment)
    plt.xlabel("Epochs")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.show()
    plt.savefig(experiment + '_bce_loss.pdf')

    fig.clf()
    fig = plt.figure(figsize=(8, 8))
    plt.plot(epochs, train_aps)
    plt.plot(epochs, val_aps)
    plt.title(experiment)
    plt.xlabel("Epochs")
    plt.ylabel("Average Precision Score")
    plt.show()
    plt.savefig(experiment + '_aps.pdf')

def plot_batch_loss(file_name, epochs, type):
    experiment = file_name.split('/')[-3]
    data = pd.read_csv(file_name)

    loss = data[type+'_loss'].values
    aps = data[type+'_aps'].values

    #fig = plt.figure(figsize=(8, 8))
    x = range(0, len(loss))
    xcoords = []
    for epoch in range(1, epochs):
        xcoords.append(epoch * (len(loss)/epochs))

    fig = plt.gcf()
    plt.plot(x, loss)
    plt.title(experiment)
    plt.xlabel("Batch")
    plt.ylabel("Binary Cross Entropy Loss")
    for xc in xcoords:
        plt.axvline(x=xc)
    #plt.show()
        fig.savefig(experiment + '_bce_loss_batch_' + type + '.pdf')

    fig.clf()
    fig = plt.gcf()
    #fig = plt.figure(figsize=(8, 8))
    plt.plot(x, aps)
    for xc in xcoords:
        plt.axvline(x=xc)
    plt.title(experiment)
    plt.xlabel("Batch")
    plt.ylabel("Average Precision Score")
    #plt.show()
    plt.savefig(experiment + '_aps_batch_' + type + '.pdf')

plot_batch_loss(file_name='experiments/standard_4_experiment/result_outputs/train_summary.csv', epochs=12, type='train')