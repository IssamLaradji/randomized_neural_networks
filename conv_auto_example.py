from sklearn.datasets import fetch_mldata
import random
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import Autoencoder, MultilayerPerceptronClassifier
from sklearn.neural_network import DBNClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.decomposition import FastICA, PCA
from sklearn.neural_network import ELMConvolution, ELMClassifier, ELMAutoEncoder
from scipy.io import savemat
from sklearn.datasets import load_digits
from cnn import ConvolutionalNeuralNetworkClassifier

import time
import random
random.seed(1)

np.random.seed(1)
for size in [500]:
    datasets = []
    mnist = fetch_mldata('MNIST original')
    X, y = mnist.data, mnist.target
    random.seed(100)
    #indices = np.array(random.sample(range(70000), size))
    #X, y = X[indices].astype('float64'), y[indices]
    X = X / 255
    # datasets.append(('MNIST',X,y))

    import issamKit as issam
    name = 'MAHDBase_TestingSet'

    X = np.load(
        'D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/samples.npy')
    y = np.array(
        np.load('D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/target.npy'))

    X = X.astype('float64')
    X = X / 255

    datasets.append(('MAHDBase', X, y))

    name = 'AHDBase_TrainingSet'

    X = np.load(
        'D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/samples.npy')
    y = np.array(
        np.load('D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/target.npy'))
    X = X / 255
    datasets.append(('AHDBase', X, y))

    for dataset in datasets:
            print "======================================================"
            np.random.seed(1)
            data_name, X, y = dataset

            X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                X, y,
                test_size=0.2,
                random_state=0)

            savemat(
                data_name + '.mat', {'X': X, 'y': y})
            continue
            # clfs = [('Logistic Regression',LogisticRegression() ), ('SVC',SVC(C=1) ),\
            #        (('ELM Convolution', ELMConvolution(n_filters = 120, kernel_size = 8))),
            #        (('ELM', ELMClassifier(n_hidden = 700)))]

            # clfs= [('DBN convolution', ConvolutionalNeuralNetworkClassifier(\
            #            max_iter = 50,n_filters=5, dim_filter = 5, dim_pool = 1)),\
            #('ELM Convolution', ELMConvolution(n_filters = 49, kernel_size = 8))]

            clfs = [('SVC', SVC(C=1)), ('ELM', ELMClassifier(n_hidden=600))]
            print 'Scores for', size, 'samples of the', data_name, 'dataset'

            for (name, clf) in []:
                s = time.time()
                score = cross_validation.cross_val_score(
                    clf, X, y, cv=5, scoring='accuracy')
                print name, 'accuracy: ', np.around(np.mean(score), 2), \
                    ' || error: ', 1 - np.around(np.mean(score), 2), \
                    ' || std: ', np.around(np.std(score), 3),\
                    ' || time:', time.time() - s
