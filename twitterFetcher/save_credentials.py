import json

# Enter your keys/secrets as strings in the following fields
credentials = {}
credentials['CONSUMER_KEY'] = 'm8ldgVv8k81bZLgiAAqOopqH6'
credentials['CONSUMER_SECRET'] = 'FPyv3OA72jY5CzZxa5Q4YuKqGoKAT0zQoAAHEFAKO73nemLuB6'
credentials['ACCESS_TOKEN'] = '2480261701-k3gjYs3RRiiZUuAI7GFMfUXpjWCB270ClYnZgCr'
credentials['ACCESS_SECRET'] = 'qsZJ09VbQBlgRhCQNFDAviLFn58BmfGRb6Qd9Xl9mGMiu'

# Save the credentials object to file
with open("twitter_credentials.json", "w") as file:
    json.dump(credentials, file)