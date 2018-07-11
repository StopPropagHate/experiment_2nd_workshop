import sys
import numpy as np
import pandas as pd

import spacy
import pt_core_news_sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem.rslp import RSLPStemmer
from sentimento import SentiLex
from textstat.textstat import *
import seaborn as sn
import pickle
from sklearn import metrics
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

stopwords = nltk.corpus.stopwords.words("portuguese")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

sentiment_analyzer = SentiLex()
stemmer = RSLPStemmer()
from sklearn import preprocessing
nlp = pt_core_news_sm.load()

class OtherTransformer(TransformerMixin, BaseEstimator):

    feature_names = ['FKRA', 'FRE', 'syllables', 'num_chars', 'num_chars_total', 'num_terms', 'num_words','num_unique_terms', 'sentiment','twitter_objs[2]', 'twitter_objs[1]' ]
    def __init__(self):
        pass
    def transform(self,X,**transform_params):
        temp=[]
        for document in X:
            features = other_features_(document[1])
            temp.append(features)

        features = preprocessing.MinMaxScaler().fit_transform(np.array(temp))
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    ## names are related to featureSetConfiguration
    def get_feature_names(self):
        return sorted(OtherTransformer.feature_names)

file = 'dataset/dataset_dummy_classes.csv'
# returns items = {key:tweet_id: (text, hate.speech) }
def load_dataset(file=file):
    df = pd.read_csv(file)
    items = df.loc[:, ['tweet_id', 'text', 'Hate.speech']]
    #5668
    return items

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    #print('id: ' ,text_string[0])
    text_string = text_string[1]

    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    #print('preprocess:', parsed_text)
    return parsed_text

def create_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=13, shuffle=True)
    X_train =  X_train.as_matrix()
    y_train = y_train.as_matrix().ravel()
    X_test = X_test.as_matrix()
    y_test = y_test.as_matrix().ravel()
    return (X_train, y_train, X_test, y_test)

def tag(tweet):
    return get_pos_tags(tweet)

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    #tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    #print('tokenize: ', tokens)
    return tokens

def basic_tokenize(tweet):

    print('(basic_tokenize ) tweet ', tweet)
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    tokens = [t for t in tweet.split()]
    print('basic_tokenize: ', tokens)
    return tokens

def get_pos_tags(tweet):
    """Takes a list of strings (tweets) and
    returns a list of strings of (POS tags).
    """
    tweet_tags = []
    doc = nlp(tweet)
    tag_list = [w.pos_ for w in doc]
    #for i in range(0, len(tokens)):
    tag_str = " ".join(tag_list)
    tweet_tags.append(tag_str)
    return tweet_tags

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def other_features_(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features.

    This is modified to only include those features in the final
    model."""
    sentiment = sentiment_analyzer.get_sentiment_tweet(tokenize(tweet))
    words = preprocess(tweet) #Get text only
    syllables = textstat.syllable_count(words) #count syllables in words
    num_chars = sum(len(w) for w in words) #num chars in words
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
    twitter_objs = count_twitter_objs(tweet) #Count #, @, and http://
    features = [FKRA, FRE, syllables, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment,
                twitter_objs[2], twitter_objs[1],]
    #features = pandas.DataFrame(features)
    return features

def build_pipeline(X,y,mode='load_from_file'):

    filename = 'pt_classifier_joblib.pkl'
    print('len X', len(X))

    if mode== 'create_model':
        print('creating model on: ', filename)
        #max_df=0.85,
        #create transformers
        vectorizer = TfidfVectorizer(analyzer='word',min_df=2 ,tokenizer=tokenize,preprocessor=preprocess,ngram_range=(1, 3),lowercase=True, stop_words=stopwords)
        other = OtherTransformer()
        pos_tagger = CountVectorizer(min_df=2,tokenizer=tag,preprocessor=preprocess,ngram_range=(1, 3))
        #create classifier
        svc = LinearSVC(random_state=0, class_weight ='balanced')
        selection =  SelectFromModel(LinearSVC(penalty="l1", dual=False), threshold=0.30)
        #create pipeline
        pipeline = Pipeline([ ("features", FeatureUnion([('vectorizer', vectorizer),('pos',pos_tagger),('other',other)])),('feature_selection',selection),('svc', svc)])
        #fit
        print('fitting..')
        pipeline.fit(X,y)

        #save model
        with open(filename,'wb+') as fo:
            print('dumping model at :',filename)
            joblib.dump(pipeline,fo)
    elif mode == 'load_from_file':
        print('loading model from: ',filename)
        with open(filename, 'rb') as fi:
            pipeline = joblib.load(fi)

    return pipeline


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    df = load_dataset()
    X = df.loc[:, ['tweet_id', 'text']]
    y = df['Hate.speech'].to_frame()
    class_counts = pd.value_counts(y.values.flatten())
    class_counts.plot.bar(x='class values', y='ammount')
    #plt.show()
    print('num instances: ', len(X),'---',len(y))
    print('creating train test split...')

    (X_train, y_train, X_test, y_test)= create_split(X,y)
    print('Build pipeline..')
    #load_from_file
    #create_model
    pipeline = build_pipeline(X_train, y_train,sys.argv[2])

    if sys.argv[1] == 'evaluation':
        print('starting evaluation..')
        # train predictions
        train_predictions = pipeline.predict(X_train)
        # performance
        datasetAccuracy = np.mean(train_predictions == y_train)
        print("Train Accuracy= " + str(datasetAccuracy))
        print('Train classification report..')
        print(classification_report(y_train, train_predictions, target_names=['No hate','Hate']))
        # Confusion Matrix table
        print("\nTrain Confusion Matrix:")
        cm_train = metrics.confusion_matrix(y_train, train_predictions)
        print(cm_train)
        df_cm = pd.DataFrame(cm_train, index=['real No','real Hate'],columns=['predicted No','predicted Hate'])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')
        plt.show()
        # test predictions
        test_predictions = pipeline.predict(X_test)
        print('Test classification report..')
        print(classification_report(y_test, test_predictions, target_names=['No hate','Hate']))
        datasetAccuracy = np.mean(test_predictions == y_test)
        print("Test Accuracy= " + str(datasetAccuracy))
        # Confusion Matrix table
        print("\nTest Confusion Matrix:")
        cm_test = metrics.confusion_matrix(y_test, test_predictions)
        print(cm_test)
        df_cm = pd.DataFrame(cm_test, index=['real No','real Hate'],columns=['predicted No','predicted Hate'])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')
        plt.show()

    print('printing model weights..')
    df = pd.DataFrame(columns=['feature_index', 'feature_name','feature_weight'])
    coefs = pipeline.named_steps['svc'].coef_
    names = pipeline.named_steps['features'].get_feature_names()
    #old 105328 # new 705
    print('features created: ', len(names))
    print('features used: ',len(coefs[0]))
    for c in np.ndenumerate(coefs):
        feature_id = c[0][1]
        feature_weight = c[1]
        df = df.set_value(feature_id, 'feature_index',feature_id)
        df = df.set_value(feature_id, 'feature_weight', feature_weight)
        for featureName in np.ndenumerate(names):
            name_id = featureName[0][0]
            if (name_id == feature_id):
                feature_name = featureName[1]
                df.set_value(feature_id, 'feature_name', feature_name)
                break

    df = df.sort_values( by=['feature_weight'])
    df.to_csv('feature_weights.csv', sep='\t', encoding='utf-8')



    print('The End')