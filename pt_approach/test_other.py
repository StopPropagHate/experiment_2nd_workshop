from classifier import count_twitter_objs
from classifier import OtherTransformer
from classifier import other_features

#load data
from classifier import load_dataset

#load data
df = load_dataset('dataset/dataset_dummy_classes.csv')
tweets = df['text']



other = OtherTransformer()

other_features = other.fit_transform(tweets)

print(other_features)