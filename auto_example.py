from sklearn.datasets import fetch_mldata
import random
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from autoencoder import Autoencoder
from dbn import DBNClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.decomposition import FastICA, PCA
from elm import ELMConvolution, ELMClassifier, ELMAutoEncoder
import time
import random

random.seed(1)
import random
random.seed(1)


def test_elm_autoencoder_MNIST():
    sizes = [500, 1000, 2000, 4000, 8000]
    for size in sizes:
        """
        mnist = fetch_mldata('MNIST original')
        X, y = mnist.data, mnist.target
        indices = np.array(random.sample(range(70000), size))
        X, y = X[indices].astype('float64'), y[indices]

        X /= 255
        """
        name = 'MAHDBase_TestingSet'
        X = np.load(
            'D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/samples.npy')[:size]
        y = np.array(
            np.load('D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/target.npy'))[:size]

        X = X.astype('float64')
        X /= 255
        s = time.time()
        clf = Autoencoder(
            algorithm='l-bfgs',
            max_iter=200,
            n_hidden=150,
            random_state=3)
        clf = ELMAutoEncoder(n_hidden=550)
        ae_features = clf.fit_transform(X)
        print 'time:', time.time() - s
        print size, 'samples'

        clfs = [(('ELM', ELMClassifier(n_hidden=700))),
                (('SVC', SVC(C=1.2)))]

        for name, clf in clfs:
            # extracted features should have higher score
            # print name,  ':', np.mean(cross_validation.cross_val_score(clf,
            # X, y, cv = 5,scoring=scoring[0]))

            score = cross_validation.cross_val_score(
                clf, X, y, cv=5, scoring='accuracy')
            print name, ': ', np.around(np.mean(score), 2), ' \pm ', np.around(np.std(score), 3)

            score = cross_validation.cross_val_score(
                clf, ae_features, y, cv=5, scoring='accuracy')
            print name, 'with SAE: ', np.around(np.mean(score), 2), ' \pm ', np.around(np.std(score), 3)
            # print name, 'with SAE:',
            # np.mean(cross_validation.cross_val_score(clf, ae_features, y,
            # cv=5, scoring=scoring[0]))
        print "================================="
test_elm_autoencoder_MNIST()


def test_DBN_vs_ELM():
    sizes = [500, 1000, 3000, 5000]
    for size in sizes:
        mnist = fetch_mldata('MNIST original')
        X, y = mnist.data, mnist.target
        indices = np.array(random.sample(range(70000), size))
        X, y = X[indices].astype('float64'), y[indices]

        X /= 255

        s = time.time()

        print size, 'samples'

        clfs = [(
            'DBN', DBNClassifier(n_hidden=[256, 100], activation='logistic', random_state=0)),
            (('ELM', ELMClassifier(n_hidden=500)))]

        for name, clf in clfs:
            # extracted features should have higher score
            if name == 'ELM':
                clf = ELMAutoEncoder(n_hidden=500)
                ae_features = clf.fit_transform(X)
                print 'name:', np.mean(cross_validation.cross_val_score(clf, ae_features, y, cv=5, scoring='accuracy'))
            else:
                print 'name:', np.mean(cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy'))


def love():

    datasets = []

    mnist = fetch_mldata('MNIST original')
    X, y = mnist.data, mnist.target
    random.seed(100)
    indices = np.array(random.sample(range(70000), 6000))
    X, y = X[indices].astype('float64'), y[indices]

    datasets.append((X, y))

    name = 'MAHDBase_TestingSet'

    X = np.load(
        'D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/samples.npy')[:5000]
    y = np.array(
        np.load('D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/target.npy')[:5000])

    X = X.astype('float64')
    #X = X / 255

    datasets.append((X, y))

    name = 'AHDBase_TrainingSet'

    X = np.load(
        'D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/samples.npy')[:5000]
    y = np.array(
        np.load('D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/target.npy')[:5000])
    X = X / 255
    datasets.append((X, y))

    # importance of feature extraction using ELM autoencoders
    for dataset in datasets:
        np.random.seed(1)
        X, y = dataset
        #X = PCA(n_components=300).fit_transform(X)
        clfs = [(
            'Logistic Regression', LogisticRegression()), ('SVC', SVC(C=1)),
            (('ELM', ELMConvolution(n_filters=20, kernel_size=8))),
            (('ELM', ELMClassifier(n_hidden=700)))]

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=0.3, random_state=0)

        for (name, clf) in clfs:
            clf.fit(X_train, y_train)

            print name, ':', accuracy_score(clf.predict(X_test), y_test)

    else:
        for dataset in datasets:
            np.random.seed(1)
            X, y = dataset
            X /= 255

            clf = ELMAutoEncoder(n_hidden=100, verbose=True, max_iter=200)
            ae_features = clf.fit_transform(X)
            #issam.display_network('D:/Datasets/ar_png/matrices/corr0corrhidden.png', clf.coef_hidden_)
            #issam.display_network('D:/Datasets/ar_png/matrices/corr0corroutput.png', clf.coef_output_.T)
            # eqweweqe
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                X, y, test_size=0.2, random_state=0)

            ae_train = fe.fit_transform(X_train)
            ae_test = fe.transform(X_test)

            clf.fit(X_train, y_train)

            print 'Raw :', accuracy_score(clf.predict(X_test), y_test)
            clf.fit(ae_train, y_train)

            print 'Autoencoder :', accuracy_score(clf.predict(ae_test), y_test)


#('DBN',DBNClassifier(n_hidden=[256, 100 ], activation='logistic', random_state=0) )
#(('MLP', MultilayerPerceptronClassifier(n_hidden = 100)))


#from scipy import signal
#from numpy.fft import fft2
#from scipy.fftpack import dct
#X = PCA(n_components=100).fit_transform(X)
#X = signal.qmf(X)[:]
