import pandas, numpy, scipy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split



def removeStopwords(df, stopword_update):
    stop = set(stopwords.words('french'))
    stop.update(stopword_update)
    df = df.applymap(lambda x: [i for i in x if i not in stop])
    return(df)

    

def stemmerWord(df):
    stemmer = SnowballStemmer('french')
    df = df.applymap(lambda x: [stemmer.stem(plural) for plural in x])
    return(df)

    

def preprocess ():
    data = pandas.read_csv('data/train.csv', sep=';')
    data_train, data_test, target_train, target_test = train_test_split(data[['ID', 'review_content', 'review_title', 'review_stars']], data[['Target']], test_size=0.2, random_state=42)
    data_train['text'] = data_train['review_title'].str.cat(data_train['review_content'], sep = ';')
    data_test['text'] = data_test['review_title'].str.cat(data_test['review_content'], sep = ';')
    data_train[['text']] = data_train[['text']].applymap(lambda x: x.lower())
    data_test[['text']] = data_test[['text']].applymap(lambda x: x.lower())

    data_train[['text']] = data_train[['text']].applymap(lambda x: x.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('à', 'a').replace('â', 'a').replace('ù', 'u').replace('û', 'u').replace('î', 'i').replace('ï', 'i').replace('ô', 'o').replace('ç', 'c'))
    data_test[['text']] = data_test[['text']].applymap(lambda x: x.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('à', 'a').replace('â', 'a').replace('ù', 'u').replace('û', 'u').replace('î', 'i').replace('ï', 'i').replace('ô', 'o').replace('ç', 'c'))


    tokenizer = RegexpTokenizer(r'\w+')  
    data_train[['text']] = data_train[['text']].applymap(lambda x: tokenizer.tokenize(x))
    data_test[['text']] = data_test[['text']].applymap(lambda x: tokenizer.tokenize(x))


    stopword_update = ['a', 'ils', 'elles', 'les', 'ca', 'cet','cette', 'cela', 'celui', 'ceque', 'si', '1', '7', 'meme', 'ete', 'etee', 'etees', 'etes', \
                       'etant', 'etante', 'etants', 'etantes', 'etes', 'etais', 'etait', 'etions', 'etiez', 'etaient', 'fumes','futes', 'eumes', 'eutes', 'eut']

    data_train[['text']] = removeStopwords(data_train[['text']], stopword_update)
    data_train[['text']] = stemmerWord(data_train[['text']])



    data_test[['text']] = removeStopwords(data_test[['text']], stopword_update)
    data_test[['text']] = stemmerWord(data_test[['text']])


    data_train[['text']] = data_train[['text']].applymap(lambda x: ','.join(x))
    #data_train[['text']].head(10)
    data_test[['text']] = data_test[['text']].applymap(lambda x: ','.join(x))
    #data_test[['text']].head(10)
    data_train[['text']].to_pickle('data/data_train.pkl')
    data_test[['text']].to_pickle('data/data_test.pkl')
    target_train.to_pickle('data/target_train.pkl')
    target_test.to_pickle('data/target_test.pkl')



