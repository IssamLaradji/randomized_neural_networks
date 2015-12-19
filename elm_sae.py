

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import pinv2
from sklearn.utils import check_random_state, atleast2d_or_csr
from sklearn.utils import as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import logistic_sigmoid, safe_sparse_dot
import scipy

class BaseELM(BaseEstimator):
    __metaclass__ = ABCMeta

    def __init__(self, n_hidden, corruption):
        #self.regressor = regressor
        self.n_hidden = n_hidden
        self.corruption = corruption
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """

    @abstractmethod
    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """

def _softmax(Z):
    """Implements the K-way softmax, (exp(Z).T / exp(Z).sum(axis=1)).T

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
    """
    exp_Z = np.exp(Z)
    return (exp_Z.T / exp_Z.sum(axis=1)).T

class ELMRegressor(BaseELM, RegressorMixin):

    def __init__(self, n_hidden = 20, corruption = 0.5):

        super(ELMRegressor, self).__init__(n_hidden, corruption)

        self.hidden_activations_ = None

    def _init_fit(self, X):
        """Initialize weight and bias parameters."""
        rng = check_random_state(0)
        self.coef_hidden_ = np.random.normal(0, 0.0005, (self.n_features, self.n_hidden))
        #self.coef_hidden_ = scipy.linalg.orth(self.coef_hidden_)
        self.intercept_hidden_ = np.random.normal(0, 0.0005, (1,self.n_hidden))
        #self.intercept_hidden_ = scipy.linalg.orth(self.intercept_hidden_.T).T


        #create orthogonal weights
        #self.coef_hidden_ = pinv2(self.coef_hidden_).T
        #self.intercept_hidden_ = safe_sparse_dot(np.identity(self.intercept_hidden_.shape[0]), self.intercept_hidden_.T)
        #print safe_sparse_dot(self.coef_hidden_.T, self.coef_hidden_)
        #add sparsity
        #self._weight_init_pca(X)
        corr = np.random.binomial(1, self.corruption, self.coef_hidden_.shape)
        indices = np.where(corr == 0)
        self.coef_hidden_[indices]=0

        corr = np.random.binomial(1,0.5,self.intercept_hidden_.shape)
        indices = np.where(corr == 0)
        self.intercept_hidden_[indices]=0
        #self.coef_hidden_ = corr
        
        self.coef_output_ = rng.uniform(-1, 1, (self.n_hidden, self.n_outputs))
        
        self.intercept_output_ = rng.uniform(-1, 1, self.n_outputs)

    def _weight_init_pca(self, X):
        from sklearn.decomposition import RandomizedPCA, SparsePCA, NMF, PCA
        pca = PCA(n_components=self.n_hidden)
        B = pca.fit_transform(X)
        #B = logistic_sigmoid(B)
        #first = safe_sparse_dot(X.T, X)

        #print pinv2(first+1*np.identity(first.shape[0])).shape, B.shape
        #self.coef_hidden_ = safe_sparse_dot(pinv2(first+1*np.identity(first.shape[0])), safe_sparse_dot(X.T, B) )
        self.coef_hidden_ = safe_sparse_dot(pinv2(X),  B)

    def _regularized(self, y):

        first = safe_sparse_dot(self.hidden_activations_.T, self.hidden_activations_)

        self.coef_output_ = safe_sparse_dot(pinv2(first+1*np.identity(first.shape[0])), safe_sparse_dot(self.hidden_activations_.T, y))

        #self.coef_output_ = scipy.linalg.orth(self.coef_output_.T).T
        #### next 

        

    def _opposite(self,X, y):
        left = pinv2(X)
        middle =  y
        right = pinv2(self.coef_output_)
        self.coef_hidden_ =  safe_sparse_dot(safe_sparse_dot(left,middle), right)

    def _fit_regression(self, y):
        self.coef_output_ = safe_sparse_dot(pinv2(self.hidden_activations_), y)

    def fit_transform(self, X, y):
        self.fit(X,y)
        return self._get_new_features(X)





    def fit(self, X, y):


        n_samples, self.n_features = X.shape
        self.n_outputs = y.shape[1]
        self._init_fit(X)
     
        self.hidden_activations_ = self._get_hidden_activations(X)

        #self._fit_regression(as_float_array(y, copy=True))
        self._regularized(as_float_array(y, copy=True))
        #self._opposite(X, as_float_array(y, copy=True))
        #self._both(X, as_float_array(y, copy=True))

        return self

    def _get_predictions(self, X):
        """get predictions using internal least squares/supplied regressor"""

        preds = safe_sparse_dot(self.hidden_activations_, self.coef_output_)

        return preds



    def _get_hidden_activations(self, X):
        A = safe_sparse_dot(X, self.coef_hidden_)
        A += self.intercept_hidden_
        Z = logistic_sigmoid(A)
        return Z

    def _get_new_features(self, X):
        A = safe_sparse_dot(X, self.coef_output_.T)
        #A += self.intercept_hidden_
        Z = logistic_sigmoid(A)
        return Z

    def predict(self, X):

      

        # compute hidden layer activations
        self.hidden_activations_ = self._get_hidden_activations(X)

        # compute output predictions for new hidden activations
        predictions = self._get_predictions(X)

        return predictions

class ELMClassifier(BaseELM, ClassifierMixin):
    def __init__(self, n_hidden = 20, corruption = 0):

        super(ELMClassifier, self).__init__(n_hidden, corruption)

        self.classes_ = None
        self.binarizer_ = LabelBinarizer(-1, 1)
        self.elm_regressor_ = ELMRegressor(n_hidden)

    def decision_function(self, X):
      
        return self.elm_regressor_.predict(X)

    def fit(self, X, y):
       
        self.classes_ = np.unique(y)

        y_bin = self.binarizer_.fit_transform(y)

        y_bin = X.copy() # to remove 

        self.elm_regressor_.fit(X, y_bin)
        return self

    def predict(self, X):
        raw_predictions = self.decision_function(X)
        class_predictions = self.binarizer_.inverse_transform(raw_predictions)

        return class_predictions
		
    def predict_proba(self, X):
        scores = self.decision_function(X)

        if len(scores.shape) == 1:
            scores = logistic_sigmoid(scores)
            return np.vstack([1 - scores, scores]).T
        else:
            return _softmax(scores)

def main():
    import issamKit as issam
    import random
    from sklearn import cross_validation
    #from sklearn.neural_network import DBNClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.datasets import fetch_mldata
    np.random.seed(0)
    clf = ELMRegressor(n_hidden=700)
    mnist = fetch_mldata('MNIST original')
    X, y = mnist.data, mnist.target
    #DBNClassifier().fit(X,y)
    indices = np.array(random.sample(range(70000), 5000))
    X, y = X[indices].astype('float64'), y[indices]
    X/=255
    #clf.fit(X, X)
    import time
    s = time.time()
    ae_features = clf.fit_transform(X, X)
    issam.display_network('D:/Datasets/ar_png/matrices/corr0corrhidden.png', clf.coef_hidden_)
    issam.display_network('D:/Datasets/ar_png/matrices/corr0corroutput.png', clf.coef_output_.T)
         
    #clf = ELMRegressor(n_hidden=500, corruption = 0.3)
    #ae_features2 = clf.fit_transform(X, X)
    print 'hell', s-time.time()
    #werer

    clf = LogisticRegression()
    #clf.fit(X, y)

    # extracted features should have higher score
    print 'SGD on raw pixels score: ', np.mean(cross_validation.cross_val_score(clf, X, y, cv = 3))

    from sklearn.neural_network import Autoencoder
    ae = Autoencoder(
        algorithm='l-bfgs',
        verbose=True,
        max_iter=400,
        n_hidden=150,
        random_state=3)

    print 'SGD on extracted Sparse auto-encoder features score: ', np.mean(cross_validation.cross_val_score(clf, ae_features, y, cv=3))
    #print 'SGD on extracted Sparse auto-encoder features score: ', np.mean(cross_validation.cross_val_score(clf, ae_features2, y, cv=3))

    #print 'SGD on extracted features score: ', np.mean(cross_validation.cross_val_score(clf, ae_features, y, cv=3))

    # extracted features should have higher score
    #print 'SGD on raw pixels score: ', clf.score(X, y)

    #issam.display_network('D:/Datasets/ar_png/matrices/corr0corr2.png', clf.coef_output_.T)
main()