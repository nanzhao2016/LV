from sklearn.feature_selection import chi2, mutual_info_classif
from time import time
import pandas
import sys, os
from start import start
import nb, lr, rvm, knn, rf, lsvm, lsvm2

    
def nb_percentile(function, filename):    
    percentiles = [5, 10, 20, 50]
    accuracys = []
    rocs = []
    for percentile in percentiles:
        accuracy, roc = nb.modeling_percentil(function, percentile)
        accuracys.append(accuracy)
        rocs.append(roc)
    df = pandas.DataFrame({'percentiles':percentiles, 'accuracys': accuracys, 'rocs': rocs})
    df.to_csv(filename)
    
def nb_kbest(function, filename):    
    K_best = [10, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 10000]
    accuracys = []
    rocs = []
    for k in K_best:
        accuracy, roc = nb.modeling_KBest(function, k)
        accuracys.append(accuracy)
        rocs.append(roc)
    df = pandas.DataFrame({'K_best':K_best, 'accuracys': accuracys, 'rocs': rocs})
    df.to_csv(filename)
    
def nb_NMF(filename):    
    n_components = [10, 100, 200, 500]
    accuracys = []
    rocs = []
    for n_component in n_components:
        accuracy, roc = nb.modeling_NMF(n_component)
        accuracys.append(accuracy)
        rocs.append(roc)
    df = pandas.DataFrame({'n_components':n_components, 'accuracys': accuracys, 'rocs': rocs})
    df.to_csv(filename)
    
if __name__ == '__main__':
    start(sys.argv[1])
    nb_percentile(mutual_info_classif, 'data/nb_percentile_mutual.csv')
    #nb_percentile(chi2, 'nb_percentile_mutual.csv')
    #nb_kbest(chi2, 'nb_kbest_chi2.csv')
    #nb_kbest(mutual_info_classif, 'nb_kbest_mutual.csv')
    #rvm.modeling_TruncatedSVD(10)
    #lr.modeling_TruncatedSVD(10)
    #lr.modeling_PCA(10)
    #knn.modeling_TruncatedSVD(10)
    #rf.modeling_TruncatedSVD(10)