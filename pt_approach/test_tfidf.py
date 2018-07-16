#load data
from classifier import load_dataset


import nltk
from classifier import preprocess
from classifier import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
###########################################

#load data
df = load_dataset('dataset/dataset_dummy_classes.csv')
tweets = df['text']

#
stopwords = nltk.corpus.stopwords.words("portuguese")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,
    min_df=5,
    max_df=0.75,
)

#Construct tfidf matrix and get relevant scores
tfidf = vectorizer.fit_transform(tweets).toarray()
vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
idf_vals = vectorizer.idf_
idf_dict = {i:idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF scores


