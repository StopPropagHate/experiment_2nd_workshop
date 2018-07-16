#load data
from classifier import load_dataset

#
import pandas as pd
from classifier import preprocess
from classifier import get_pos_tags
from sklearn.feature_extraction.text import TfidfVectorizer



#load data
df = load_dataset('dataset/dataset_dummy_classes.csv')
tweets = df['text']


#Get POS tags for tweets and save as a string
tweet_tags = []
for t in tweets:
    clean = preprocess(t)
    tags =get_pos_tags(clean)
    tag_str = " ".join(tags)
    tweet_tags.append(tag_str)


#We can use the TFIDF vectorizer to get a token matrix for the POS tags
pos_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None,
    use_idf=False,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.75,
    )



#Construct POS TF matrix and get vocab dict
pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}
print(pos_vocab)