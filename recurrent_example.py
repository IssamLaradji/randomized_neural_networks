import csv
import os
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
import sklearn.linear_model as lm
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MultilayerPerceptronRegressor, ELMRegressor
from pandas import read_csv
loadData = lambda f: np.genfromtxt(open(f, 'r'), delimiter=',')
import datetime
import numpy as np
import pylab as pl
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import matplotlib.pyplot as plt
from sklearn import preprocessing


loc = 'E:/Datasets/recurrent_datasets/'

dataset_names = ['Short-movement Stock Prices', 'Yahoo Stock Values',
                 '1D wind data', '4D wind data']

chaos_names = ['Chaos dataset A', 'Chaos dataset B',
               'Chaos dataset C', 'Chaos dataset D',
               'Chaos dataset E', 'Chaos dataset F']
line_styles = ['--', '-', '-.']
line_markers = ['*', 'o', 'x']


def get_recurrent_datasets():
    datasets = []

    # stock dataset
    X = np.empty((200, 198))
    y = np.empty((200, 198))
    labels = loadData(loc + 'trainLabels.csv')
    for i, data_no in enumerate(np.arange(200) + 1):
        temp_data = loadData(loc + 'data/' + str(data_no) + '.csv')
        temp_X, temp_y = temp_data[1:, :], temp_data[1:, :198]
        X[i] = temp_y[-1]
        y[i] = labels[i + 1][1:]
    X = preprocessing.MinMaxScaler().fit_transform(X)

    np.save('X.npy', X)
    np.save('y.npy', y)

    print
    datasets.append((X, y))

    # yahoo dataset
    date1 = datetime.date(1991, 1, 1)  # start date
    date2 = datetime.date(2012, 1, 6)  # end date
    # get quotes from yahoo finance
    quotes = quotes_historical_yahoo("INTC", date1, date2)
    # Downloading the data
    date1 = datetime.date(1991, 1, 1)  # start date
    date2 = datetime.date(2012, 1, 6)  # end date
    # get quotes from yahoo finance
    quotes = quotes_historical_yahoo("INTC", date1, date2)
    if len(quotes) == 0:
        raise SystemExit

    # unpack quotes
    dates = np.array([q[0] for q in quotes], dtype=int)
    close_v = np.array([q[2] for q in quotes])
    volume = np.array([q[2] for q in quotes])[1:]

    # take diff of close value
    # this makes len(diff) = len(close_t) - 1
    # therefore, others quantity also need to be shifted
    diff = close_v[1:] - close_v[:-1]
    dates = dates[1:]
    close_v = close_v[1:]

    # pack diff and volume for training
    #X = np.column_stack([diff, volume, dates - 726835])
    X = np.column_stack([volume])
    X = X[:, np.newaxis]
    #X = np.array(read_csv('recc_data.csv'))[:20000]

    y = X[1:]
    X = X[:-1]

    datasets.append((X, y))

    # 1D Wind data
    X = np.array(read_csv(loc + '1DWind_Data.csv'))[:20000, :1]
    y = X[1:]
    X = X[:-1]
    timesteps = np.arange(len(X))[:, np.newaxis]
    #X = np.hstack([X, timesteps])
    datasets.append((X, y))

    # 4D Wind data
    X = np.array(read_csv(loc + '4DWind_Data.csv'))[:20000]
    y = X[1:]
    X = X[:-1]

    datasets.append((X, y))

    return datasets


def get_wind_dataset():
    datasets = []
    # 4D Wind data
    X = np.array(read_csv(loc + '4DWind_Data.csv'))[:20000]
    y = X[1:]
    X = X[:-1]

    datasets.append((X, y))

    return datasets


def get_chaos_datasets():
    datasets = []
    dataset_names = ['A', 'B', 'C', 'D', 'E', 'F']

    for name in dataset_names:
        filename = loc + name + '.dat'
        count = 0

        for line in open(filename, 'r'):
            features = line.rstrip().split()
            count += 1

        X = np.zeros((count, len(features)))
        for i, line in enumerate(open(filename, 'r')):
            vector = line.rstrip().split()

            vector = [v if v != 'NA' else 0 for v in vector]
            X[i] = np.array(vector)

        y = X[1:]
        X = X[:-1]
        # print name+' Samples #:', X.shape[0], 'Feauteres #:', X.shape[1]
        datasets.append((X, y))

    return datasets


def start():
    datasets = get_wind_dataset()

    index = 0
    the_name = '4D wind data'
    print the_name

    X, y = datasets[index]
    reps = 10
    n_hiddens = range(2, 8)
    results_nonrecurrent = []
    results_recurrent = []
    results_sequential = []
    results_names = [
        'Non-Recurrent ELM', 'Recurrent Hidden ELM', 'Recurrent Output ELM']
    results = [[], [], []]
    cv = True

    random_samples = [0, 1, 5, 1, 6, 6]

    for j, n_hidden in enumerate(n_hiddens):

        clfs = [ELMRegressor(
            n_hidden=n_hidden), ELMRegressor(
                algorithm='recurrent', n_hidden=n_hidden),
            ELMRegressor(algorithm='recurrent_sequential', n_hidden=n_hidden)]
        # ten fold cross validation
        for step, clf in enumerate(clfs):
            mean_auc = 0.0
            n = reps
            # default np.random.seed(1)
            # dataset E, F np.random.seed(6)
            np.random.seed(random_samples[index])
            for i in range(n):
                X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                    X, y, test_size=0.2, random_state=i * 25)

                clf.fit(X_train, y_train)
                preds = clf.predict(X_cv, (X_train, y_train))
                if step == 1:
                    X_cv, y_cv = X_cv[1:], y_cv[1:]
                sort_index = np.argsort(X_cv[:, -1])

                mae = metrics.mean_absolute_error(y_cv[:, 0], preds[:, 0])
                mean_auc += mae

            avg_mae = mean_auc / n
            results[step].append(avg_mae)

        print "hidden neurons: ", n_hidden
        for i in range(len(results_names)):
            print "Mean MAE for " + results_names[i] + ":" + str(results[i][j])

    pl.title(the_name)
    pl.ylabel('Mean Absolute Error')
    pl.xlabel('Hidden Neurons')

    for i in range(len(results_names)):
        pl.plot(
            n_hiddens, results[i], marker=line_markers[
                i], linestyle=line_styles[i],
            label=results_names[i])

    pl.legend()
    pl.savefig('E:/thesis_images/' + the_name + '.png', format='png')
    pl.close()
    # pl.show()
start()
werwer


def plot_them():
    datasets = get_recurrent_datasets()
    #X,y =  datasets[0]
    for index, (X, y) in enumerate(datasets):
        pl.ylabel('Value')
        pl.title(dataset_names[index])
        pl.xlabel('Time Step')
        print X.shape[1]
        for i in range(X.shape[1]):
            pl.plot(np.arange(X.shape[0]), X[:, i])
        pl.savefig('E:/thesis_images/' + dataset_names[
                   index] + '_plot.png', format='png')
        pl.close()


def main():
    cv = True

    data = np.array(read_csv(loc + 'recc_data.csv'))[:20000]

    X = np.load('X.npy')
    y = np.load('y.npy')

    timesteps = np.arange(len(X))[:, np.newaxis]
    X = np.hstack([X, timesteps])

    cut = int(0.8 * max(timesteps))
    #ch2 = SelectKBest(chi2, 300)
    #X=ch2.fit_transform(X, y)
    print 'data extraction complete'
    clf = ELMRegressor()
    # ten fold cross validation
    if cv:
        print "Cross Validating..."
        mean_auc = 0.0
        n = 1
        for i in range(n):
            #X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=0.2, random_state=i*25)
            X_train, X_cv, y_train, y_cv = X[:cut], X[cut:], y[:cut], y[cut:]
            clf.fit(X_train[:, :-1], y_train[:, :-1])
            preds = clf.predict(
                X_cv[:, :-1], (X_train[:, :-1], y_train[:, :-1]))
            #X_cv, y_cv = X_cv[1:], y_cv[1:]
            sort_index = np.argsort(X_cv[:, -1])
            # print y_cv[:10,1], X_cv[sort_index,2]
            plt.plot(X_cv[sort_index, -1], y_cv[sort_index, 8])
            plt.plot(X_cv[sort_index, -1], preds[sort_index, 8])
            plt.ylabel('Stock Value')
            plt.xlabel('Time steps')
            plt.show()

            mae = metrics.mean_absolute_error(y_cv[:, :-1], preds)
            print "MAE (fold %d/%d): %f" % (i + 1, n, mae)
            mean_auc += mae
        print "Mean MAE: %f" % (mean_auc / n)


def main_yahoo():

    date1 = datetime.date(1991, 1, 1)  # start date
    date2 = datetime.date(2012, 1, 6)  # end date
    # get quotes from yahoo finance
    quotes = quotes_historical_yahoo("INTC", date1, date2)
    # Downloading the data
    date1 = datetime.date(1991, 1, 1)  # start date
    date2 = datetime.date(2012, 1, 6)  # end date
    # get quotes from yahoo finance
    quotes = quotes_historical_yahoo("INTC", date1, date2)
    if len(quotes) == 0:
        raise SystemExit

    # unpack quotes
    dates = np.array([q[0] for q in quotes], dtype=int)
    close_v = np.array([q[2] for q in quotes])
    volume = np.array([q[2] for q in quotes])[1:]

    # take diff of close value
    # this makes len(diff) = len(close_t) - 1
    # therefore, others quantity also need to be shifted
    diff = close_v[1:] - close_v[:-1]
    dates = dates[1:]
    close_v = close_v[1:]

    # pack diff and volume for training
    X = np.column_stack([diff, volume, dates - 726835])

    #X = np.array(read_csv('recc_data.csv'))[:20000]

    y = X[1:]
    X = X[:-1]
    cv = True

    cut = int(0.8 * X.shape[0])
    #ch2 = SelectKBest(chi2, 300)
    #X=ch2.fit_transform(X, y)
    print 'data extraction complete'
    clf = ELMRegressor(n_hidden=5)
    # ten fold cross validation
    if cv:
        print "Cross Validating..."
        mean_auc = 0.0
        np.random.seed(2)
        n = 10
        for i in range(n):
            X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                X, y, test_size=0.2, random_state=i * 25)
            #X_train, X_cv, y_train, y_cv = X[:cut], X[cut:],y[:cut], y[cut:]
            clf.fit(X_train, y_train)
            preds = clf.predict(X_cv, (X_train, y_train))
            #X_cv, y_cv = X_cv[1:], y_cv[1:]
            sort_index = np.argsort(X_cv[:, -1])
            # print y_cv[:10,1], X_cv[sort_index,2]

            # plt.plot(X_cv[sort_index,-1],y_cv[sort_index,1])
            # plt.plot(X_cv[sort_index,-1],preds[sort_index,1])
            #plt.ylabel('Stock Value')
            #plt.xlabel('Time steps')
            # plt.show()

            mae = metrics.mean_absolute_error(y_cv[:, 1], preds[:, 1])
            # print "MAE (fold %d/%d): %f" % (i + 1, n, mae)
            mean_auc += mae

    non_recurrent_mae = mean_auc / n

    clf = ELMRegressor(algorithm='recurrent', n_hidden=5)
    # ten fold cross validation
    if cv:
        print "Cross Validating..."
        mean_auc = 0.0
        np.random.seed(0)
        n = 10
        for i in range(n):
            X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                X, y, test_size=0.2, random_state=i * 25)
            #X_train, X_cv, y_train, y_cv = X[:cut], X[cut:],y[:cut], y[cut:]
            clf.fit(X_train, y_train)
            preds = clf.predict(X_cv, (X_train, y_train))
            X_cv, y_cv = X_cv[1:], y_cv[1:]
            sort_index = np.argsort(X_cv[:, -1])
            # print y_cv[:10,1], X_cv[sort_index,2]

            # plt.plot(X_cv[sort_index,-1],y_cv[sort_index,1])
            # plt.plot(X_cv[sort_index,-1],preds[sort_index,1])
            #plt.ylabel('Stock Value')
            #plt.xlabel('Time steps')
            # plt.show()

            mae = metrics.mean_absolute_error(y_cv[:, 1], preds[:, 1])
            # print "MAE (fold %d/%d): %f" % (i + 1, n, mae)
            mean_auc += mae

        recurrent_mae = mean_auc / n

    clf = ELMRegressor(algorithm='recurrent_sequential', n_hidden=3)
    # ten fold cross validation
    if cv:
        print "Cross Validating..."
        mean_auc = 0.0
        np.random.seed(1)
        n = 10
        for i in range(n):
            #X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=0.2, random_state=i*25)
            X_train, X_cv, y_train, y_cv = X[:cut], X[cut:], y[:cut], y[cut:]
            clf.fit(X_train, y_train)
            preds = clf.predict(X_cv, (X_train, y_train))
            #X_cv, y_cv = X_cv[1:], y_cv[1:]
            sort_index = np.argsort(X_cv[:, -1])
            # print y_cv[:10,1], X_cv[sort_index,2]

            # plt.plot(X_cv[sort_index,-1],y_cv[sort_index,1])
            # plt.plot(X_cv[sort_index,-1],preds[sort_index,1])
            #plt.ylabel('Stock Value')
            #plt.xlabel('Time steps')
            # plt.show()

            mae = metrics.mean_absolute_error(y_cv[:, 1], preds[:, 1])
            # print "MAE (fold %d/%d): %f" % (i + 1, n, mae)
            mean_auc += mae

        sequential_recurrent_mae = mean_auc / n

        print "Mean MAE for non-recurrent ELM: %f" % non_recurrent_mae
        print "Mean MAE for recurrent ELM: %f" % recurrent_mae
        print "Mean MAE for sequential recurrent ELM: %f" % sequential_recurrent_mae


def main_wind():

    X = np.array(read_csv(loc + 'Wind_Data.csv'))[:20000]
    #X = np.array(read_csv('recc_data.csv'))[:20000]

    y = X[1:]
    X = X[:-1]
    cv = True

    cut = int(0.8 * X.shape[0])
    #ch2 = SelectKBest(chi2, 300)
    #X=ch2.fit_transform(X, y)
    print 'data extraction complete'
    reps = 10
    n_hiddens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results_nonrecurrent = []
    results_recurrent = []
    results_sequential = []
    results_svr = []

    for n_hidden in n_hiddens:
        clf = ELMRegressor(n_hidden=n_hidden)
        # ten fold cross validation
        if cv:
            # print "Cross Validating..."
            mean_auc = 0.0
            np.random.seed(1)
            n = reps
            for i in range(n):
                X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                    X, y, test_size=0.2, random_state=i * 25)
                #X_train, X_cv, y_train, y_cv = X[:cut], X[cut:],y[:cut], y[cut:]
                clf.fit(X_train, y_train)
                preds = clf.predict(X_cv, (X_train, y_train))
                #X_cv, y_cv = X_cv[1:], y_cv[1:]
                sort_index = np.argsort(X_cv[:, -1])
                # print y_cv[:10,1], X_cv[sort_index,2]

                # plt.plot(X_cv[sort_index,-1],y_cv[sort_index,1])
                # plt.plot(X_cv[sort_index,-1],preds[sort_index,1])
                #plt.ylabel('Stock Value')
                #plt.xlabel('Time steps')
                # plt.show()

                mae = metrics.mean_absolute_error(y_cv[:, 0], preds[:, 0])
                # print "MAE (fold %d/%d): %f" % (i + 1, n, mae)
                mean_auc += mae

        non_recurrent_mae = mean_auc / n
        results_nonrecurrent.append(non_recurrent_mae)
        clf = ELMRegressor(algorithm='recurrent', n_hidden=n_hidden)
        # ten fold cross validation
        if cv:
            # print "Cross Validating..."
            mean_auc = 0.0
            np.random.seed(1)
            n = reps
            for i in range(n):
                X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                    X, y, test_size=0.2, random_state=i * 25)
                #X_train, X_cv, y_train, y_cv = X[:cut], X[cut:],y[:cut], y[cut:]
                clf.fit(X_train, y_train)
                preds = clf.predict(X_cv, (X_train, y_train))
                X_cv, y_cv = X_cv[1:], y_cv[1:]
                sort_index = np.argsort(X_cv[:, -1])
                # print y_cv[:10,1], X_cv[sort_index,2]

                # plt.plot(X_cv[sort_index,-1],y_cv[sort_index,1])
                # plt.plot(X_cv[sort_index,-1],preds[sort_index,1])
                #plt.ylabel('Stock Value')
                #plt.xlabel('Time steps')
                # plt.show()

                mae = metrics.mean_absolute_error(y_cv[:, 0], preds[:, 0])
                # print "MAE (fold %d/%d): %f" % (i + 1, n, mae)
                mean_auc += mae

            recurrent_mae = mean_auc / n
            results_recurrent.append(recurrent_mae)

        clf = ELMRegressor(
            algorithm='recurrent_sequential', n_hidden=n_hidden)
        # ten fold cross validation
        if cv:
            # print "Cross Validating..."
            mean_auc = 0.0
            np.random.seed(1)
            n = reps
            for i in range(n):
                X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                    X, y, test_size=0.2, random_state=i * 25)
                #X_train, X_cv, y_train, y_cv = X[:cut], X[cut:],y[:cut], y[cut:]
                clf.fit(X_train, y_train)
                preds = clf.predict(X_cv, (X_train, y_train))
                #X_cv, y_cv = X_cv[1:], y_cv[1:]
                sort_index = np.argsort(X_cv[:, -1])
                # print y_cv[:10,1], X_cv[sort_index,2]

                # plt.plot(X_cv[sort_index,-1],y_cv[sort_index,1])
                # plt.plot(X_cv[sort_index,-1],preds[sort_index,1])
                #plt.ylabel('Stock Value')
                #plt.xlabel('Time steps')
                # plt.show()

                mae = metrics.mean_absolute_error(y_cv[:, 0], preds[:, 0])
                # print "MAE (fold %d/%d): %f" % (i + 1, n, mae)
                mean_auc += mae

            sequential_recurrent_mae = mean_auc / n
            results_sequential.append(sequential_recurrent_mae)

            print "hidden neurons: ", n_hidden
            print "Mean MAE for non-recurrent ELM: %f" % non_recurrent_mae
            print "Mean MAE for recurrent ELM: %f" % recurrent_mae
            print "Mean MAE for sequential recurrent ELM: %f" % sequential_recurrent_mae
        """
        clf=SVR()
        #ten fold cross validation
        if cv:
            #print "Cross Validating..."
            mean_auc = 0.0
            np.random.seed(1)
            n = reps
            for i in range(n):
                X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=0.2, random_state=i*25)
                #X_train, X_cv, y_train, y_cv = X[:cut], X[cut:],y[:cut], y[cut:]
                
                clf.fit(X_train, y_train[:, 0]) 
                
                preds = clf.predict(X_cv)
                #X_cv, y_cv = X_cv[1:], y_cv[1:]
                sort_index = np.argsort(X_cv[:,-1])
                #print y_cv[:10,1], X_cv[sort_index,2] 

                #plt.plot(X_cv[sort_index,-1],y_cv[sort_index,1])
                #plt.plot(X_cv[sort_index,-1],preds[sort_index,1])
                #plt.ylabel('Stock Value')
                #plt.xlabel('Time steps')
                #plt.show()

                mae = metrics.mean_absolute_error(y_cv[:,0],preds[:])
                #print "MAE (fold %d/%d): %f" % (i + 1, n, mae)
                mean_auc += mae

            svr_mae = mean_auc/n
            results_svr.append(svr_mae)
            """

            # print "Mean MAE for Support Vector Machines: %f" % svr_mae
    pl.title('Japan 4D Wind data')
    pl.plot(n_hiddens, results_nonrecurrent, marker='o',
            linestyle='--', label="Non-Recurrent ELM")
    pl.plot(n_hiddens, results_recurrent, marker='o',
            linestyle='-', label="Recurrent Hidden ELM")
    pl.plot(n_hiddens, results_sequential,
            marker='o', label="Recurrent Output ELM")
    pl.ylabel('Mean Absolute Error')
    pl.xlabel('Hidden Neurons')
    pl.legend()
    # pl.show()
    pl.savefig('E:/thesis_images/' + save + '.png', format='png')
    pl.close()

if __name__ == "__main__":
    # main()
    # main_yahoo()
    # start()
    # start()
    plot_them()
