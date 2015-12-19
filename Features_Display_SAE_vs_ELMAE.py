from elm import ELMAutoEncoder
import autoencoder
import numpy as np
import random
from autoencoder import Autoencoder
import random
from sklearn.datasets import fetch_mldata


def main():
    np.random.seed(0)
    datasets = []
    clfs = [('ELMAE', ELMAutoEncoder(n_hidden=100)), ('SAE', Autoencoder(
            max_iter=200,
            n_hidden=100))]

    mnist = fetch_mldata('MNIST original')
    X = mnist.data
    indices = np.array(random.sample(range(70000), 1000))
    X = X[indices].astype('float64')

    datasets.append(('MNIST', X))

    name = 'MAHDBase_TestingSet'

    X = np.load(
        'D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/samples.npy')[:3000]

    X = X.astype('float64')
    X = X / 255

    datasets.append(('MAHD', X))

    name = 'AHDBase_TrainingSet'

    X = np.load(
        'D:/Dropbox/Arabic_Font_Stuff/Datasets/' + name + '/samples.npy')[:1000]

    datasets.append(('AHDBase', X))

    for dataset_name, X in datasets:
        for clf_name, clf in clfs:
            name = dataset_name + '_' + clf_name + '_features.png'
            clf.fit(X)
            if clf_name == 'SAE':
                issam.display_network(
                    name,
                    clf.coef_hidden_)
            else:
                issam.display_network(
                    name,
                    clf.coef_output_.T)

            print name, 'done'

main()
