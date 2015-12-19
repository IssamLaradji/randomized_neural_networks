
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import pinv2, cholesky, pinv

from sklearn.utils import check_random_state, atleast2d_or_csr
from sklearn.utils import as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import logistic_sigmoid, safe_sparse_dot
from sklearn.utils import gen_even_slices
import scipy
from scipy import sparse
from numpy.linalg import lstsq
from scipy import optimize
from sklearn import cross_validation, metrics
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from numpy.fft import fft2, ifft2,rfft2,irfft2
from scipy.signal import convolve2d, convolve

from scipy.linalg import sqrtm, inv
import random
random.seed(1)

def RandomMatrixFromEigenvalues(E):
     n    = len(E)
     # Create a random orthogonal matrix Q via QR decomposition
     # of a random matrix A
     A    = np.mat(np.random.random((n,n)))
     Q, R = np.linalg.qr(A)
     #  similarity transformation with orthogonal
     #  matrix leaves eigenvalues intact
     return Q


def sym(w):
    return w.dot(inv(sqrtm(w.T.dot(w))))

def _centered(ret, size):
    # Return the center newsize portion of the array.
    start = (ret.shape - size) // 2
    end = start + size
    return ret[start[0]:end[0], start[1]:end[1]]

def myconvolve(image, kernel, mode="valid"):
    n = np.array(image.shape)
    m = np.array(kernel.shape)
    size = n + m - 1
    ret = ifft2(fft2(image, size) * fft2(kernel, size)).real
    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, n)
    elif mode == "valid":
        return _centered(ret, n - m + 1)


def norm_shape(shape):
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
 
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass
     
    raise TypeError('shape must be an int, or a tuple of ints')

from numpy.lib.stride_tricks import as_strided as ast

def jacobian(x,sign=1.):
        return sign*(np.dot(x.T,H) + c)

class BaseELM(BaseEstimator):
    __metaclass__ = ABCMeta

    def __init__(self, n_hidden, algorithm, kernel, adaptive, random_state):
        #self.regressor = regressor
        self.n_hidden = n_hidden
        self.kernel = kernel
        self.algorithm = algorithm
        self.adaptive = adaptive
        self.random_state = random_state

    def _init_fit(self, X):
        """Initialize weight and bias parameters."""
        rng = check_random_state(self.random_state)

        r  = np.sqrt(6) / np.sqrt(self.n_outputs+self.n_hidden + 1);
       
        self.coef_hidden_ = rng.rand(self.n_features, self.n_hidden) * 2 * r - r;
        self.intercept_hidden_ = rng.rand(self.n_hidden) * 2 * r - r; 

        # raises ValueError if not registered
        if self.algorithm not in ["regular", "weighted", "recurrent", "sequential", "recurrent_sequential", "regularized", "QP", 'seq feature']:
            raise ValueError("The algorithm %s "
                             " is not supported. " % self.algorithm)
        if self.kernel not in ["random", "rbf", "poly", "wave"]:
            raise ValueError("The kernel %s"
                             " is not supported. " % self.kernel)


        #self.coef_hidden_ =  np.array([[ -0.31178367, 0.72900392], [0.21782079, -0.8990918]])
        #self.intercept_hidden_ = np.array([ -2.48678065, 0.91325152 ])

        #self.coef_output_ = rng.uniform(-1, 1, (self.n_hidden, self.n_outputs))
        #self.intercept_output_ = rng.uniform(-1, 1, self.n_outputs)

    #F = (1/2)*x.T*P*x + q.T*x

    def _partial_init_fit(self, X, n_features):

        rng = check_random_state(self.random_state)

        n_new_features = n_features - 2

        r  = np.sqrt(6) / np.sqrt(self.n_outputs+ n_new_features + 1);

        W = rng.rand(n_features, n_new_features) * 2 * r - r;
        b = rng.rand(n_new_features) * 2 * r - r; 

        return W, b


    def _get_partial_hidden_activations(self, X, W, b):

        A = safe_sparse_dot(X, W)
        A += b

        Z = np.tanh(A)

        return Z

    def _forward_selection(self, X, y):

            X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                  X, y, test_size=0.3, random_state=0)

            n_samples, self.n_features = X_train.shape
            self.X_train = X_train
          
            self.n_outputs = y.shape[1]

            
            indices = []
            start = 0
            old_output = 0
            Start = True
            total = 0
            increment_size = 16
            while  total < X_train.shape[1]:
                
                total += increment_size
                indices = range(start, start+increment_size)
                n_features = len(indices)
                start += increment_size
                W,b = self._partial_init_fit(X_train[:,indices], n_features)

                if Start:
                    Start = False
                   
                    old_H = self._get_partial_hidden_activations(X_train[:,indices] ,W,b)
                    old_H_pinv = pinv2(old_H)
                    test_old_H = self._get_partial_hidden_activations(X_test[:,indices],W,b)

                    self.coef_output_ = safe_sparse_dot(old_H_pinv, y_train)

                else:
                    new_H = self._get_partial_hidden_activations(X_train[:,indices] ,W,b)
                    new_test_H = self._get_partial_hidden_activations(X_test[:,indices],W,b)
                    test_H = np.hstack([test_old_H, new_test_H])
                    
                    A = (np.eye(old_H.shape[0])-safe_sparse_dot(old_H, old_H_pinv))
                    D = pinv2(safe_sparse_dot(A,new_H))
                    #print new_H.T.shape, D.shape
                    #print safe_sparse_dot(new_H.T, D)
                    U = safe_sparse_dot(old_H_pinv, (np.eye(old_H.shape[0]) - safe_sparse_dot(new_H, D) ))

                    #print np.vstack([U,D]).shape, y_train.shape

                    self.coef_output_ = safe_sparse_dot(np.vstack([U,D]), y_train)

                    old_H = np.hstack([old_H, new_H])
                    old_H_pinv = pinv2(old_H)
                    test_old_H = test_H.copy()

                y_pred =  safe_sparse_dot(test_old_H, self.coef_output_ )
                y_pred = self.binarizer_.inverse_transform(y_pred)

                score = accuracy_score( y_pred, self.binarizer_.inverse_transform(y_test))
                print str(total)+' features :' + str(score)
                """
                total += increment_size
                indices = range(start, start+increment_size)
                start += increment_size

                n_features = len(indices)
                W,b = self._partial_init_fit(X_train[:,indices], n_features)

                new_hidden_activations = self._get_partial_hidden_activations(X_train[:,indices] ,W,b)
                new_hidden_activations_test = self._get_partial_hidden_activations(X_test[:,indices],W,b)

                #new_beta = safe_sparse_dot(pinv2(new_hidden_activations), y_train - old_output)
                new_beta = safe_sparse_dot(pinv2(new_hidden_activations), y_train - old_output)
               
                #first = safe_sparse_dot(
                        #new_hidden_activations.T, new_hidden_activations)
                #new_beta = safe_sparse_dot(
                         #pinv2(first + 1 * np.identity(first.shape[0])), safe_sparse_dot(
                         #new_hidden_activations.T, y_train))
              
                #old + new
                if Start:
                    self.coef_output_ = new_beta
                    hidden_activations_ = new_hidden_activations
                    self.coef_hidden_ = W
                    self.intercept_hidden_ = b
                    hidden_activations_test_ = new_hidden_activations_test

                else:
                    if total == X_train.shape[1]:

                        hidden_activations_test_ = np.hstack([hidden_activations_test_, new_hidden_activations_test])
                        self.coef_output_ = np.vstack([self.coef_output_, new_beta])
                    else:
                        self.coef_output_ = np.vstack([self.coef_output_, new_beta])
                        hidden_activations_test_ = np.hstack([hidden_activations_test_, new_hidden_activations_test])

                    hidden_activations_ = np.hstack([hidden_activations_, new_hidden_activations]) 
                    self.coef_hidden_ = np.vstack([self.coef_hidden_, W])
                    self.intercept_hidden_ = np.vstack([self.intercept_hidden_, b])
                    
                  
                # * (float(increment_size))/total

                old_output = safe_sparse_dot(hidden_activations_, self.coef_output_)

                y_pred =  safe_sparse_dot(hidden_activations_test_, self.coef_output_ )
                y_pred = self.binarizer_.inverse_transform(y_pred)

                score = accuracy_score( y_pred, self.binarizer_.inverse_transform(y_test))

                #auc = (metrics.mean_absolute_error( self.binarizer_.inverse_transform(y_test), y_pred))

                #score = self.score(X_test[:,range(total)], self.binarizer_.inverse_transform(y_test))

                print str(total)+' features :' + str(score)
                Start = False
                """
            return self


    def _get_hidden_activations(self, X):

        if self.kernel == 'poly':
            c, d = 0, 2

            return self._kernel_poly(X, c ,d)

        elif self.kernel == 'rbf':
            sig2 = 600

            return self._kernel_rbf(X, sig2)

        elif self.kernel =='wav':
            return self._kernel_wav(X)

        A = safe_sparse_dot(X, self.coef_hidden_)
        A += self.intercept_hidden_

        Z = np.tanh(A)

        return Z

    def _optimize(self, X, y):
        n_samples, n_features = X.shape
 
        K = self._get_hidden_activations(X)
        print K.shape
        # Gx <= h
        cons1 = lambda x: h - np.dot(G,x)
        # Ax = b
        cons2 = lambda x: np.dot(A.T,x)
        #print x.shape, b.shape, y.shape
        P = np.outer(y,y) * K
        q = np.ones(n_samples) * -1
        A = y
        b = 0.0
        G = np.eye(n_samples) * -1
        h = np.zeros(n_samples)
        x0 = np.random.random(n_samples)

        def objective(x):
            
            return 1.*(0.5*np.dot(x.T,np.dot(P,x))+ np.dot(q.T,x))

        cons = ({'type':'ineq', 'fun':cons1,
                 'type':'eq', 'fun':cons2 })
        res_cons = optimize.minimize(objective, x0, constraints=cons,
            method='SLSQP',options={'disp':False})
        return res_cons






    def _kernel_wav(self, X, a=1, b=1, c=1):
        XXh1 = np.sum(self.X_train**2,1)[:,np.newaxis].dot(np.ones((1, X.shape[0])));
        XXh2 = np.sum(X**2, 1)[:,np.newaxis].dot(np.ones((1,self.X_train.shape[0])));
        omega = XXh1+XXh2.T - 2*(self.X_train.dot(X.T))
        
        XXh11 = np.sum(self.X_train,1)[:,np.newaxis].dot(np.ones((1,X.shape[0])))
        XXh22 = np.sum(X, 1)[:,np.newaxis].dot(np.ones((1,self.X_train.shape[0])))
        omega1 = XXh11-XXh22.T
        
        omega = np.cos(c*omega1/b)*np.exp(-omega/a);

        return omega.T

    def _adaptive_hidden_neurons():
        pass

    def _kernel_poly(self, X, c, d):

        if self.adaptive:

                if not hasattr(self, 'power'): self.power = 1

                if self.status == 'training':

                    if not hasattr(self, 'old_value_train'): self.old_value_train = np.power((X.dot(self.X_train.T)+c), self.power)
                

                    self.old_value_train = np.power(self.old_value_train, (float(self.power+1.)/self.power))

                    return self.old_value_train

                elif self.status == 'testing':

                    if not hasattr(self, 'old_value_test'): self.old_value_test = np.power((X.dot(self.X_train.T)+c), self.power)

                    self.old_value_test = np.power(self.old_value_test, (float(self.power+1.)/self.power))

                    self.power = self.power  + 1.
        
                    return self.old_value_test



        return np.power((X.dot(self.X_train.T)+c), d)

    def _kernel_rbf(self, X, sig2):

        """

        XXh1 = np.sum(self.X_train**2, 1)[:,np.newaxis].dot(np.ones((1,X.shape[0])));
        XXh2 = np.sum(X**2,1)[:,np.newaxis].dot(np.ones((1,self.X_train.shape[0])));
        omega = XXh1+XXh2.T - 2*self.X_train.dot(X.T);
        omega = np.exp(-omega/sig2);

        return omega.T
        """

        n_samples = X.shape[0]

        def gaussian_kernel(x, y, sigma=5.0):
            return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):

            for j in range(n_samples):

                K[i,j] = gaussian_kernel(X[i], X[j])

        return K


 

    def _solve_regularized(self, y):
        first = safe_sparse_dot(
            self.hidden_activations_.T, self.hidden_activations_)
        self.coef_output_ = safe_sparse_dot(
            pinv2(first + 1 * np.identity(first.shape[0])), safe_sparse_dot(
                self.hidden_activations_.T, y))

    def _solve_weighted(self, y):

        n_samples, n_features = self.hidden_activations_.shape

        w = np.ones(n_samples)

        classes = np.unique(y)
        weights = {-1:1, 1:1000}

        for class_ in classes:
            indices = np.where(y==class_)[0]
            w[indices] = 1./indices.size
            #w[indices] = weights[class_]


        W = np.identity(n_samples)*w


        C = 32

        H_W = safe_sparse_dot(
            self.hidden_activations_.T, W )

        term = safe_sparse_dot(
            H_W, self.hidden_activations_ )

        #self.coef_output_ = lstsq(term + (np.identity(n_features)/C),  safe_sparse_dot(
               #H_W, y))[0]

        self.coef_output_ = pinv2(term + (np.identity(n_features)/C)).dot(safe_sparse_dot(
                H_W, y))


        
    
    def _solve(self, y):


        if self.adaptive and self.kernel =='random':

            if not hasattr(self, 'old_beta'): 
            
                self.old_beta = safe_sparse_dot(pinv2(self.hidden_activations_), y)

            else:

                self.old_beta = safe_sparse_dot(pinv2(self.hidden_activations_), y)

            self.coef_output_ = self.old_beta

        else:
   
            self.coef_output_ = safe_sparse_dot(pinv2(self.hidden_activations_), y)
            #self.coef_output_ = lstsq(self.hidden_activations_, y)[0]

    def _sequential_solve(self, X, y):

            n_samples = X.shape[0]
            outputs = y.shape[1]
            
            batch_size = np.clip(1000, 0, n_samples)
            n_batches = n_samples / batch_size
            batch_slices = list(
                gen_even_slices(
                    n_batches * batch_size,
                    n_batches))

            K, Beta = np.zeros((self.n_hidden, self.n_hidden)), \
                np.zeros((self.n_hidden, outputs))

            for batch_slice in batch_slices:
                    K, Beta = self._sequential_solve_procedure(
                        X[batch_slice],
                        y[batch_slice],
                        K,
                        Beta)

            self.coef_output_ = Beta

    def _sequential_solve_procedure(self, X_batch, y_batch, old_K, old_beta):
        from numpy.linalg import inv
        H = self._get_hidden_activations(X_batch)

        K = old_K + (H.T).dot(H)
        #print (y_batch - H.dot(old_beta) )
        Beta = old_beta + lstsq(K, (H.T).dot((y_batch - H.dot(old_beta))))[0]
        #Beta = old_beta + (inv(K).dot(H.T)).dot((y_batch - H.dot(old_beta)))
        return K, Beta


    #######################


    def _get_last_output(self, X, last_output):
        X = atleast2d_or_csr(X)
        H = self._get_hidden_activations(X)
        H = np.hstack([H, last_output])

       
        output = safe_sparse_dot(H, self.coef_output_)

        return output

    def _sequential_solve_recurrent(self, X, y):

            n_samples = X.shape[0]
            self.outputs = y.shape[1]
            outputs = y.shape[1]
            # for dataset chaos A batch_size = np.clip(50, 0, n_samples)
            batch_size = np.clip(500, 0, n_samples)
            n_batches = n_samples / batch_size
            batch_slices = list(
                gen_even_slices(
                    n_batches * batch_size,
                    n_batches))

            K, Beta, last_output = np.zeros((self.n_hidden+ outputs, self.n_hidden+ outputs)), \
                np.zeros((self.n_hidden + outputs, outputs)),\
                np.zeros((batch_size, outputs))

            
            for i, batch_slice in enumerate(batch_slices):
                    if i == 0:
                        last_X_batch = X[batch_slice]

                        clf = ELMRegressor()
                        clf.fit(X,y)

                        last_output = clf.predict(X[batch_slice])
                        continue
                    K, Beta, last_output = self._sequential_solve_recurrent_procedure(
                        X[batch_slice],
                        y[batch_slice],
                        last_X_batch,
                        K,
                        Beta,
                        last_output)
                    last_X_batch = X[batch_slice]

            self.coef_output_ = Beta

    def _sequential_solve_recurrent_procedure(self, X_batch, y_batch, last_X_batch, old_K, old_beta, last_output):
        from numpy.linalg import inv
        H = self._get_hidden_activations(X_batch)
        #print 'werwe', last_output.shape
        H = np.hstack([H, last_output])

        K = old_K + (H.T).dot(H)
        #print (y_batch - H.dot(old_beta) )
        #print H.shape, old_beta.shape
        #werwer
        Beta = old_beta + lstsq(K, (H.T).dot((y_batch - H.dot(old_beta))))[0]
        #Beta = old_beta + (inv(K).dot(H.T)).dot((y_batch - H.dot(old_beta)))
        self.coef_output_ = Beta
        last_output = self._get_last_output(last_X_batch, last_output)
        return K, Beta, last_output

    #######################

    def fit(self, X, y):

        if self.algorithm == 'mine':
            self._forward_selection(X,y)
            return self

        if self.adaptive:

            X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                  X, y, test_size=0.3, random_state=0)

            n_samples, self.n_features = X_train.shape
            self.X_train = X_train
          
            self.n_outputs = y.shape[1]
            self._init_fit(X_train)

            old_score = 0
            score = 0.1

            while score > old_score:
                self.status = 'training'
                old_score = score

                self.hidden_activations_ = self._get_hidden_activations(X_train)

                self._solve(as_float_array(y_train, copy=True))

                self.status = 'testing'

                score = self.score(X_test, self.binarizer_.inverse_transform(y_test))

                print score

            return self


        else: 
            n_samples, self.n_features = X.shape
            self.X_train = X
            self.n_outputs = y.shape[1]
            self._init_fit(X)
     

            if self.algorithm == 'QP':
                x = self._optimize(X, y)['x']
                return self


            if self.algorithm == 'sequential':
                self._sequential_solve(X, y)

                return self

            self.hidden_activations_ = self._get_hidden_activations(X)

            if self.algorithm == 'recurrent':
                H = self.hidden_activations_[1:]
               
                H_p = self._get_hidden_activations(X)[:-1]
                I = np.random.normal(-0.0005, 0.0005, (H.shape[1], H.shape[1]))
               
                self.hidden_activations_ = H + H_p.dot(I)

                #regularized solution
                first = safe_sparse_dot(
                    self.hidden_activations_.T, self.hidden_activations_)
                self.coef_output_ = safe_sparse_dot(
                    pinv2(first + 0.5 * np.identity(first.shape[0])), safe_sparse_dot(
                    self.hidden_activations_.T, y[1:]))
                #self._solve(as_float_array(y[1:], copy=True))

            elif self.algorithm == 'regularized':
                self._solve_regularized(as_float_array(y, copy=True))

            elif self.algorithm == 'weighted':
                self._solve_weighted(y)


            elif self.algorithm == 'recurrent_sequential':
                self._sequential_solve_recurrent(X, y)

            else:
                self._solve(as_float_array(y, copy=True))

            return self

    def decision_function(self, X, TRAINING = None):

        X = atleast2d_or_csr(X)

        # compute hidden layer activations
        if self.algorithm == 'recurrent':
                self.hidden_activations_ = self._get_hidden_activations(X)
                H = self.hidden_activations_[1:]
               
                H_p = self._get_hidden_activations(X)[:-1]
                I = np.random.normal(-0.0005, 0.0005, (H.shape[1], H.shape[1]))
               
                self.hidden_activations_ = H + H_p.dot(I)


        elif self.algorithm == 'recurrent_sequential':
                n_samples = X.shape[0]
                output = np.zeros((n_samples,self.outputs))
                batch_size = np.clip(100, 0, n_samples)
                n_batches = n_samples / batch_size
                batch_slices = list(
                    gen_even_slices(
                        n_batches * batch_size,
                        n_batches))

                for i, batch_slice in enumerate(batch_slices):
                    if i == 0:
                        X1,y1 = TRAINING
                        clf = ELMRegressor()
                        clf.fit(X1,y1)

                        last_output = clf.predict(X[batch_slice])
                        output[: batch_size] = last_output
                        continue
                    H = self._get_hidden_activations(X[batch_slice])
                    H = np.hstack([H, last_output])
                    output[i*batch_size:i*batch_size + batch_size] = safe_sparse_dot(H, self.coef_output_)
                    
                    last_output = output[i*batch_size:i*batch_size + batch_size]

                return output
        else: 
            self.hidden_activations_ = self._get_hidden_activations(X)


        output = safe_sparse_dot(self.hidden_activations_, self.coef_output_)

        return output


class ELMRegressor(BaseELM, RegressorMixin):

    def __init__(self, n_hidden=20, algorithm = "regular", kernel = "random", adaptive = False, random_state = None):

        super(ELMRegressor, self).__init__(n_hidden, algorithm, kernel, adaptive, random_state)

        self.hidden_activations_ = None

    def fit(self, X, y):

        super(ELMRegressor, self).fit(X, y)

        return self

    def predict(self, X, TRAINING = None):

        return self.decision_function(X, TRAINING)


class ELMClassifier(BaseELM, ClassifierMixin):

    def __init__(self, n_hidden=20, algorithm = "regular", kernel = "random", adaptive = False, random_state = None):

        super(ELMClassifier, self).__init__(n_hidden, algorithm, kernel, adaptive, random_state)

        self.binarizer_ = LabelBinarizer(-1, 1)
        self.classes_ = None

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        y = self.binarizer_ .fit_transform(y)

        super(ELMClassifier, self).fit(X, y)
        return self

    def predict(self, X):
        X = atleast2d_or_csr(X)
        scores = self.decision_function(X)

        # if len(scores.shape) == 1:
        #scores = logistic_sigmoid(scores)
        #results = (scores > 0.5).astype(np.int)

        # else:
            #scores = _softmax(scores)
            #results = scores.argmax(axis=1)
            # self.classes_[results]
        return self.binarizer_.inverse_transform(scores)

    def predict_proba(self, X):
        scores = self.decision_function(X)

        if len(scores.shape) == 1:
            scores = logistic_sigmoid(scores)
            return np.vstack([1 - scores, scores]).T
        else:
            return _softmax(scores)



class ELMAutoEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, n_hidden = 20, corruption = 0.5):

        self.corruption = corruption
        self.n_hidden = n_hidden
        self.hidden_activations_ = None

    def _init_fit(self, X):
        """Initialize weight and bias parameters."""
        rng = check_random_state(0)
        self.coef_hidden_ = np.random.normal(0, 0.0005, (self.n_features, self.n_hidden))
        self.intercept_hidden_ = np.random.normal(0, 0.0005, (1,self.n_hidden))
        corr = np.random.binomial(1, self.corruption, self.coef_hidden_.shape)
        indices = np.where(corr == 0)
        #self.coef_hidden_[indices]=0
        corr = np.random.binomial(1,0.5,self.intercept_hidden_.shape)
        indices = np.where(corr == 0)
        #self.intercept_hidden_[indices]=0

        if self.coef_hidden_.shape[0] > self.coef_hidden_.shape[1]:
                Q, R = scipy.linalg.qr(self.coef_hidden_)

        else:
                Q, R = scipy.linalg.qr(self.coef_hidden_.T)
                
        self.coef_hidden_ = Q[:self.coef_hidden_.shape[0], :self.coef_hidden_.shape[1]]


        self.coef_output_ = rng.uniform(-1, 1, (self.n_hidden, self.n_outputs))
        self.intercept_output_ = rng.uniform(-1, 1, self.n_outputs)

    def _regularized(self, X):

        first = safe_sparse_dot(self.hidden_activations_.T, self.hidden_activations_)
        self.coef_output_ = safe_sparse_dot(pinv2(first+1*np.identity(first.shape[0])), safe_sparse_dot(self.hidden_activations_.T, X))

    def fit_transform(self, X):
        self.fit(X)
        return self._get_new_features(X)

    def transform(self, X):

        return self._get_new_features(X)

    def fit(self, X):


        n_samples, self.n_features = X.shape
        self.n_outputs = X.shape[1]
        self._init_fit(X)
     
        self.hidden_activations_ = self._get_hidden_activations(X)

        self._regularized(as_float_array(X, copy=True))
        #self.coef_output_ = safe_sparse_dot(pinv2(self.hidden_activations_), X)

        return self

    def _get_predictions(self):
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
        A += self.intercept_hidden_
        Z = np.tanh(A)
        return Z

    def predict(self, X):
        # compute hidden layer activations
        self.hidden_activations_ = self._get_hidden_activations(X)

        predictions = self._get_predictions()

        return predictions

class ELMRBM(BaseEstimator, TransformerMixin):
    def nearPSD(self, A,epsilon=0):
       n = A.shape[0]
       eigval, eigvec = np.linalg.eig(A)
       val = np.matrix(np.maximum(eigval,epsilon))
       vec = np.matrix(eigvec)
       T = 1/(np.multiply(vec,vec) * val.T)
       T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
       B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
       out = B*B.T
       return(out)
    def _sample_hiddens(self, v, rng):
        """Sample from the distribution P(h|v)."""
        p = self._mean_hiddens(v)
        p[rng.uniform(size=p.shape) < p] = 1.
        return np.floor(p, p)

    def _sample_visibles(self, h, rng):
        """Sample from the distribution P(v|h)."""
        p = logistic_sigmoid(np.dot(h, self.components_)
                             + self.intercept_visible_)
        p[rng.uniform(size=p.shape) < p] = 1.
        return np.floor(p, p)

    def __init__(self, n_hidden = 20, corruption = 0.5):
        self.corruption = corruption
        self.n_hidden = n_hidden
        self.hidden_activations_ = None

    def _init_fit(self, X):
        """Initialize weight and bias parameters."""
        rng = check_random_state(0)
        self.coef_hidden_ = np.random.normal(0, 0.0005, (self.n_features, self.n_hidden))
        self.intercept_hidden_ = np.random.normal(0, 0.0005, (1,self.n_hidden))
        corr = np.random.binomial(1, self.corruption, self.coef_hidden_.shape)
        indices = np.where(corr == 0)
        self.coef_hidden_[indices]=0
        corr = np.random.binomial(1,0.5,self.intercept_hidden_.shape)
        indices = np.where(corr == 0)
        self.intercept_hidden_[indices]=0
        self.coef_output_ = rng.uniform(-1, 1, (self.n_hidden, self.n_outputs))
        self.intercept_output_ = rng.uniform(-1, 1, self.n_outputs)

    def _regularized(self, X):

        first = safe_sparse_dot(self.hidden_activations_.T, self.hidden_activations_)
        self.coef_output_ = safe_sparse_dot(pinv2(first+1*np.identity(first.shape[0])), safe_sparse_dot(self.hidden_activations_.T, X))

    def fit_transform(self, X):
        self.fit(X)
        M = safe_sparse_dot(pinv2(X), X)
        
        d = np.linalg.eigvalsh(M.dot(M.T))
        print d
        W = np.linalg.cholesky(d)
        print W
        #print W.shape, X.shape
        result = safe_sparse_dot(X, W)
        print result
        #print result.shape
        return result

    def fit(self, X):


        n_samples, self.n_features = X.shape
        self.n_outputs = X.shape[1]
        self._init_fit(X)
     
        self.hidden_activations_ = self._get_hidden_activations(X)

        self._regularized(as_float_array(X, copy=True))

        return self

    def _get_predictions(self):
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
        A += self.intercept_hidden_
        Z = np.tanh(A)
        return Z

    def predict(self, X):
        # compute hidden layer activations
        self.hidden_activations_ = self._get_hidden_activations(X)

        predictions = self._get_predictions()

        return predictions


class ELMConvolution(BaseEstimator, TransformerMixin):
    def __init__(self, n_filters = 20, kernel_size = 8, corruption = 0.5):
        self.corruption = corruption
        self.n_filters = n_filters
        self.hidden_activations_ = None
        self.kernel_size = kernel_size

        
        self.binarizer_ = LabelBinarizer(-1, 1)
        self.classes_ = None

    def easy_sliding_window(self, a,ws,ss = None,flatten = True):         
        if None is ss:
            # ss was not provided. the windows will not overlap in any direction.
            ss = ws
        ws = norm_shape(ws)
        ss = norm_shape(ss)
         
        # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
        # dimension at once.
        ws = np.array(ws)
        ss = np.array(ss)
        shape = np.array(a.shape)
         
         
        # ensure that ws, ss, and a.shape all have the same number of dimensions
        ls = [len(shape),len(ws),len(ss)]
        if 1 != len(set(ls)):
            raise ValueError(\
            'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
         
        # ensure that ws is smaller than a in every dimension
        if np.any(ws > shape):
            raise ValueError(\
            'ws cannot be larger than a in any dimension.\
             a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
         
        # how many slices will there be in each dimension?
        newshape = norm_shape(((shape - ws) // ss) + 1)
        # the shape of the strided array will be the number of slices in each dimension
        # plus the shape of the window (tuple addition)
        newshape += norm_shape(ws)
        # the strides tuple will be the array's strides multiplied by step size, plus
        # the array's strides (tuple addition)
        newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
        strided = ast(a,shape = newshape,strides = newstrides)
        if not flatten:
            return strided
         
        # Collapse strided so that it has one more dimension than the window.  I.e.,
        # the new array is a flat list of slices.
        meat = len(ws) if ws.shape else 0
        firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
        dim = firstdim + (newshape[-meat:])
        # remove any dimensions with size 1
        dim = filter(lambda i : i != 1,dim)
        return strided.reshape(dim)

    def sliding_window(self, imgs, ws, ss = (1,1)):
      result = None
      for a in imgs:
        ws = norm_shape(ws)
        ss = norm_shape(ss)

        ws = np.array(ws)
        ss = np.array(ss)
        shape = np.array(a.shape)
        newshape = norm_shape(((shape - ws) // ss) + 1)
        newshape += norm_shape(ws)
        newstrides = norm_shape(np.array(a.strides) * ss) + a.strides

        strided = ast(a,shape = newshape,strides = newstrides)

        meat = len(ws) if ws.shape else 0
        firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
        dim = firstdim + (newshape[-meat:])
        # remove any dimensions with size 1
        dim = filter(lambda i : i != 1,dim)
        tmp = strided.reshape(dim)

        res = tmp.reshape((-1, self.kernel_size**2))
        self.hidden_size = res.shape[0] 

        if result == None:
            result =  np.zeros((self.hidden_size * self.n_samples,self.kernel_size**2))
            start = 0

        end = start + self.hidden_size
        result[start: end] = res
        start = end
     
        #result = np.vstack([result, np.array([b.ravel().tolist() for b in tmp])])
      return result


    def get_kernels(self, beta):
        return np.array([b.reshape((self.kernel_size, self.kernel_size)) for b in beta])


    def _regularized(self, X):

        first = safe_sparse_dot(self.hidden_activations_.T, self.hidden_activations_)
        self.coef_output_ = safe_sparse_dot(pinv2(first+1*np.identity(first.shape[0])), safe_sparse_dot(self.hidden_activations_.T, X))


    def solve_convolutional_layer(self, X):

        H = logistic_sigmoid(X.dot(self.coef_input_))

        """
        beta = pinv2(H).dot(X)
        first = np.dot(H.T, H)
        self.coef_hidden_ = np.dot(pinv2(first+1*np.identity(first.shape[0])), 
                                   np.dot(H.T, X))
        """
        self.coef_hidden_ = np.dot(pinv2(H), X)
        #print self.coef_hidden_.shape
        #import issamKit as issam
        #issam.display_network('E:/corr0corrhidden.png', self.coef_hidden_)
        #issam.display_network('E:/corr0corrhiddenT.png', self.coef_hidden_.T)



    def make_pool(self, H_):

        self.my_pool_size = H_.shape[3] / self.pool_size

        new_H_ = np.zeros((self.n_samples, self.n_filters, self.my_pool_size, self.my_pool_size))
        for n in range(self.n_samples):
                for m in range(self.n_filters):
                    new_H_[n, m] = np.reshape(np.array([np.mean(i) for i in self.easy_sliding_window(H_[n,m],(self.pool_size, self.pool_size))]), \
                              (self.my_pool_size, self.my_pool_size))
        return new_H_

    def solve_output_layer(self, X, y):
        H_ = logistic_sigmoid(X.dot(self.coef_hidden_.T))

        dim = int(np.sqrt(self.hidden_size))

        if self.pool:
            H_ = np.reshape(H_, (self.n_samples, self.n_filters, dim, dim))
            H_ = self.make_pool(H_)
            H_ = np.reshape(H_, (self.n_samples, self.my_pool_size * self.my_pool_size * self.n_filters))

        else: 
            H_ = np.reshape(H_, (self.n_samples, dim * dim * self.n_filters))

        """
        beta = pinv2(H_).dot(y)
        first = np.dot(H_.T, H_)
        self.coef_output_  = np.dot(pinv2(first+1*np.identity(first.shape[0])), 
                                   np.dot(H_.T, y))
        """

        self.coef_output_ = np.dot(pinv2(H_), y)

        return H_
        #y = binarizer_.fit_transform(y)


    def cnn_convolve(self, X):
       

        self.filter_size = self.img_size - self.kernel_size + 1;

        convolved_features = np.zeros((self.n_samples, self.n_filters, self.filter_size, self.filter_size));

        for image_num in range(self.n_samples):
          for filter_num in range(self.n_filters):

            convolved_image = np.zeros((self.filter_size, self.filter_size));

            convolved_image = convolve2d(X[image_num], self.coef_input_[filter_num], 'valid')
            convolved_image = logistic_sigmoid(convolved_image)          
            convolved_features[image_num, filter_num] = convolved_image;


        return convolved_features


    def make_it_H(self, convolved_features):

        hidden_features = np.zeros((self.img_size*self.n_samples, self.img_size*self.n_filters))
        start_row = 0
        for image_num in range(self.n_samples):
          end_row = start_row +  self.img_size
          start_column = 0
          for filter_num in range(self.n_filters):
            end_column = start_column + self.img_size
            hidden_features[start_row: end_row, start_column: end_column] = \
                    fft2(convolved_features[image_num, filter_num], (self.img_size, self.img_size))

            
            start_column = end_column
          start_row = end_row
        return hidden_features

    def make_it_X(self, X):

        hidden_features = np.zeros((self.img_size*self.n_samples, self.img_size*self.n_filters))
        start_row = 0
        for image_num in range(self.n_samples):
          end_row = start_row +  self.img_size
          start_column = 0
          for filter_num in range(self.n_filters):
            end_column = start_column + self.img_size
            hidden_features[start_row: end_row, start_column: end_column] = \
                    fft2(convolved_features[image_num, filter_num], (self.img_size, self.img_size))
                    
            
            start_column = end_column
          start_row = end_row
        return hidden_features


    def fit(self, X, y):

        self.pool = False
        fft = False
        self.n_samples, self.img_size = X.shape[0], int(np.sqrt(X.shape[1]))
        y = self.binarizer_.fit_transform(y)

        X = atleast2d_or_csr(X)
        X = np.reshape(X,(-1, self.img_size, self.img_size))

       

        if fft:
            self.coef_input_ = np.random.uniform(
                -0.0005, 0.0005, (self.n_filters, self.kernel_size, self.kernel_size))

            H = self.cnn_convolve(X)
            H_ = self.make_it_H(H)

            print H_.shape

            self.coef_hidden_ = np.dot(pinv2(H_), X)

            print  self.coef_hidden_

        else:

            X = self.sliding_window(X, (self.kernel_size,self.kernel_size))
            #print 'sliding done'
            self.coef_input_ = np.random.uniform(
                #-0.0005, 0.0005, (self.kernel_size**2, self.n_filters))
                -0.0001, 0.0001, (self.kernel_size**2, self.n_filters))
            #print self.coef_input_.shape
            if self.coef_input_.shape[0] > self.coef_input_.shape[1]:
                Q, R = scipy.linalg.qr(self.coef_input_)

            else:
                Q, R = scipy.linalg.qr(self.coef_input_.T)
                
            #self.coef_input_ = Q[:self.coef_input_.shape[0], :self.coef_input_.shape[1]]

            
            
            

            corr = np.random.binomial(1, 0.3, (self.kernel_size**2, self.n_filters))
            indices = np.where(corr == 0)
            self.coef_input_[indices]=0
            self.pool_size = 3
            self.solve_convolutional_layer(X)
            #print 'convolution done'
           
            H_ = self.solve_output_layer(X, y)
            #print 'solution done'
            #y_pred = H_.dot(self.coef_output_)

        return self

    def predict(self, X):
        # compute hidden layer activations
        self.n_samples, self.img_size = X.shape[0], int(np.sqrt(X.shape[1]))
        X = atleast2d_or_csr(X)

        X = np.reshape(X,(-1, self.img_size, self.img_size))
        X = self.sliding_window(X, (self.kernel_size,self.kernel_size))
        
        #print 'test: sliding done'

        H_ = logistic_sigmoid(X.dot(self.coef_hidden_.T))
        dim = int(np.sqrt(self.hidden_size))

        if self.pool:
            H_ = np.reshape(H_, (self.n_samples, self.n_filters, dim, dim))
            H_ = self.make_pool(H_)
            H_ = np.reshape(H_, (self.n_samples, self.my_pool_size * self.my_pool_size * self.n_filters))

        else: 
            H_ = np.reshape(H_, (self.n_samples, dim * dim * self.n_filters))
        
        #H_ = np.reshape(H_, (self.n_samples, dim * dim * self.n_filters))


        return self.binarizer_.inverse_transform(H_.dot(self.coef_output_))
