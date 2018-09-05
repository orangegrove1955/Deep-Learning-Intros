import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch data and format
data = fetch_movielens(min_rating=4.0)

# Print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# Create model
model = LightFM(loss='warp')

# Fit model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):

    # Number of users and movies in training data
    n_users, n_items = data['train'].shape

    # Generate recommendations for each user as input
    for user_id in user_ids:

        # Movies already liked
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # Movies predicted to be liked
        scores = model.predict(user_id, np.arange(n_items))

        # Rank in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # Display results
        print("User %s" % user_id)
        print("     Known Positives:")
        for x in known_positives[:3]:
            print("          %s" % x)

        print("     Recommended:")
        for x in top_items[:3]:
            print("          %s" % x)

sample_recommendation(model, data, [3, 25, 450])

