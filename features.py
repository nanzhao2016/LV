import pandas, numpy, scipy, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, mutual_info_classif, SelectKBest
from sklearn.decomposition import TruncatedSVD, NMF

    
def get_train_test():
    file_handler = open('data/data_train.pkl', "rb")
    data_train = pickle.load(file_handler)
    file_handler.close()
    file_handler = open('data/data_test.pkl', "rb")
    data_test = pickle.load(file_handler)
    file_handler.close()
    file_handler = open('data/target_train.pkl', "rb")
    target_train = pickle.load(file_handler)
    file_handler.close()
    file_handler = open('data/target_test.pkl', "rb")
    target_test = pickle.load(file_handler)
    file_handler.close()
    
    target_train = target_train.Target.as_matrix()
    target_test = target_test.Target.as_matrix()
    
    vectorizer = TfidfVectorizer(analyzer='word', max_df=0.5, sublinear_tf=True)
    data_train_transformed = vectorizer.fit_transform(data_train['text'].tolist())
    data_test_transformed = vectorizer.transform(data_test['text'].tolist())
    
    print('The sparse demision of the training dataset is ', data_train_transformed.shape)
    return(data_train_transformed, data_test_transformed, target_train, target_test)


#function option: chi2 and multual_info_classif
def feature_select_Percentile(function, percentile):
    data_train_transformed, data_test_transformed, target_train, target_test = get_train_test()
    selector = SelectPercentile(function, percentile=percentile)
    selector.fit(data_train_transformed, target_train)
    data_train_transformed = selector.transform(data_train_transformed).toarray()
    data_test_transformed  = selector.transform(data_test_transformed).toarray()
    print('The filtered training dataset is ', data_train_transformed.shape)
    return(data_train_transformed, data_test_transformed, target_train, target_test)

#function option: chi2 and multual_info_classif
def feature_select_KBest(function, k):
    data_train_transformed, data_test_transformed, target_train, target_test = get_train_test()
    selector = SelectKBest(function, k=k)
    selector.fit(data_train_transformed, target_train)
    data_train_transformed = selector.transform(data_train_transformed).toarray()
    data_test_transformed  = selector.transform(data_test_transformed).toarray()
    print('The filtered training dataset is ', data_train_transformed.shape)
    return(data_train_transformed, data_test_transformed, target_train, target_test)

def feature_select_NMF(n_components):
    data_train_transformed, data_test_transformed, target_train, target_test = get_train_test()
    selector = NMF(n_components=n_components, random_state=42)
    selector.fit(data_train_transformed)
    data_train_transformed = selector.transform(data_train_transformed)
    data_test_transformed  = selector.transform(data_test_transformed)
    print('The filtered training dataset is ', data_train_transformed.shape)
    return(data_train_transformed, data_test_transformed, target_train, target_test)

def feature_select_TruncatedSVD(n_components):
    data_train_transformed, data_test_transformed, target_train, target_test = get_train_test()
    selector = TruncatedSVD(n_components=n_components, random_state=42)
    selector.fit(data_train_transformed)
    data_train_transformed = selector.transform(data_train_transformed)
    data_test_transformed  = selector.transform(data_test_transformed)
    print('The filtered training dataset is ', data_train_transformed.shape)
    return(data_train_transformed, data_test_transformed, target_train, target_test)

