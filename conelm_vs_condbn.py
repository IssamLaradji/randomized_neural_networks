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
from elm import ELMConvolution, ELMClassifier, ELMAutoEncoder
from scipy import io
mnist = fetch_mldata('MNIST original')
X, y = mnist.data, mnist.target
indices = np.array(random.sample(range(70000), 8000))
X, y = X[indices].astype('float64'), y[indices]

X/=255

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                X, y, test_size=0.2, random_state=0)
"""
X_train = X_train.reshape((-1,1,28,28))
X_test = X_test.reshape((-1,1,28,28))

X_train = X_train.transpose()
X_test = X_test.transpose()
print X_train.shape


io.savemat('stlTrainSubset.mat', {'trainImages':X_train, 'trainLabels':y_train,\
                                   'numTrainImages':y_train.shape[0]})

io.savemat('stlTestSubset.mat', {'testImages':X_test, 'testLabels':y_test,\
                                   'numTestImages':y_test.shape[0]})
"""
clf = ELMConvolution(n_filters = 49, kernel_size = 8)
clf.fit(X_train, y_train)

print accuracy_score(clf.predict(X_test), y_test) 
