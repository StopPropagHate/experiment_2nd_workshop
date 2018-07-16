from classifier import load_dataset

df = load_dataset('dataset/dataset_dummy_classes.csv')
tweets = df['text']

print(tweets)
