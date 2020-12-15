#!/usr/bin/env python
# coding: utf-8


import re
import praw
import config
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns



"""

# Predicting the Forum of origin for text data 

---

This notebook shows how to pull [Reddit](https://reddit.com/) 
posts using [PRAW](https://praw.readthedocs.io/en/latest/) in order to 

train a language model that can predict the forum a post originates from.
"""

# List of subreddits to use as a basis for our model
forums = [
    'astrology',
    'datascience',
    'machinelearning',
    'physics',
    'conspiracy'
]

# Data Splitting
TEST_SIZE = .2
RANDOM_STATE = 0
# Data preprocessing
MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER_SVD = 30
# Model parameters
N_NEIGHBORS = 4
CV = 3


# url_params = {
#     content:"submission",
#     subreddit:"",
#     after:"60s",
#     before:"0s",
#     sort_type:"score",
#     sort_how:'desc',
#     size:1000,
#     features:["created_utc","selftext", "title", "score","subreddit"]
# }

# # Format url for specific query
# def get_url(**kwargs):
    
#     url = ('https://api.pushshift.io/reddit/search/'\
#         + f'{content}/?q={topic}'\
#         + f'&after={after}&before={before}' \
#         + f'&sort_type={sort_type}&sort={sort_how}&size={size}'\
#         + f'&fields={",".join(features)}')
        
#     return url


# Load data
def load_data():
    """### Load data
    - Define list of subreddits
    - Acquire text data from each subreddit
    - Filter posts with less than 100 alphabetic characters"""

    print("Querying for 1000 most recent posts in:")
    
    for forum in forums:

        print(f"r/{forum}")


    # Create reddit object
    reddit = praw.Reddit(
        client_id = config.id,
        client_secret = config.secret,
        user_agent = 'Reddit Scraper'
    )

    # Count number of alphabetic characters using RegEx substitution
    char_count = lambda post: len(re.sub(r'\W|\d', '', post.selftext))

    # Condition for filtering the posts
    mask = lambda post: char_count(post) >= 100

    # Lists to hold results
    data = []
    labels = []

    for i, forum in enumerate(forums):

        # Get latest posts from the subreddit
        subreddit_data = reddit.subreddit(forum).new(limit=1000)

        # Filter out posts not satisfying condition
        posts = [post.selftext for post in filter(mask, subreddit_data)]

        # Add posts and labels to respective lists
        data.extend(posts)
        labels.extend([i] * len(posts))

        print(f"Number of posts from r/{forum}: {len(posts)}",
              f"\nSample Post: {posts[0][:600]}...\n",
              "_" * 80 + '\n')
    
    return data, labels



# Partition training and validation data

#     Hold out some portion of our data in order to
#     evaluate how our model performs on unseen text.

#     It is important to do this **as early as possible**

#     Keep all information about the test set
#     away from the training data if you want to accurately measure
#     how your model performs out in the wild

def split_data():
    """
    The following parameters allow for our results to be reproduced

    - TEST_SIZE: 
        percentage of total data to be held out from training set
        in order to validate our model's performance on unseen data

    - RANDOM_STATE:
        an integer corresponding to the random state of our 
        train_test_split function, and can be used to 
        reproduce the data partitions if needed 
    """

    print(f"Partitioning {100*TEST_SIZE}% of data for model evaluation...")
    # Partition training/validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels,
        test_size = TEST_SIZE,
        random_state = RANDOM_STATE)

    print(f"{len(y_test)} samples selected.")
    
    return X_train, X_test, y_train, y_test



# ### Preprocessing and feature extraction 

# - Remove symbols, numbers, and url-like strings with custom preprocessor 
# - vectorize text using term frequency-inverse document frequency
# - reduce to principal values using singular value decomposition
# - Partition data and labels into training/validation sets

def preprocessing_pipeline():

    """Tune the following parameters for efficiency/accuracy

        - MIN_DOC_FREQ: 
            minimum frequency for a term to be used in our model

        - N_COMPONENTS: 
            number of components (words) used in our vectorization,
            larger values may result in higher accuracy, 
            but increases computation time
        - N_ITER:
            number of iterations for the principal component analysis,
            larger values may result in higher accuracy, 
            but increases computation time
        """
    
    # Remove non alphabetic characters and URL-like strings
    pattern = r'\W|\d|http.*\s+|www.*\s+'
    preprocessor = lambda text: re.sub(pattern, ' ', text)

    # tf-idf vectorizer with custom preprocessing function
    vectorizer = TfidfVectorizer(
        preprocessor = preprocessor, 
        stop_words = 'english',
        min_df = MIN_DOC_FREQ)

    # SVD object to combine with vectorizer for latent semantic analysis
    decomposition = TruncatedSVD(
        n_components = N_COMPONENTS,
        n_iter = N_ITER_SVD)
    
    pipeline = [('tfidf',vectorizer),
                ('svd',decomposition)]

    return pipeline



# ### Model selection
# Chose a model or set of models that fit your use case.
# 
# Here, we've selected three distinct classification models
# 
# - KNeighborsClassifier, 
# - RandomForestClassifier, 
# - and LogisticRegressionCV

def load_models():
    """Set the following model parameters

        For 'KNeighborsClassifier'
            N_NEIGHBORS: number of neighbors to be used when predicting

        For 'LogisticRegressionCV'
            CV: number of cross-validations to make when fitting
    """

    model_1 = KNeighborsClassifier(n_neighbors = N_NEIGHBORS)
    model_2 = RandomForestClassifier(random_state = RANDOM_STATE)
    model_3 = LogisticRegressionCV(cv = CV, random_state = RANDOM_STATE)

    models = [("kneighbors", model_1),
              ("forest", model_2),
              ("lrcv", model_3)]
    
    return models



# ### Fit and Evaluate models
# 
# Fit models and evaluate performance on validation set

def fit_and_evaluate(models, pipeline, X_train, X_test, y_train, y_test):
    """This function accomplished the following
        
        - Creates a data pipeline using our preprocessing pipeline
        - Transforms corpus of raw text into Tf-idf term-document matrix
        - Fits a model to the training partition
        - Makes predictions on the validation data
        - Saves the model, predictions, and other metadata

    """
  
    results = []
  
    for name, model in models:

        # Pipeline to ensure no information from test set leaks into model
        pipe = Pipeline(pipeline + [(name, model)])

        # Fit to training data
        print(f"Fitting {name} to training data...")
        pipe.fit(X_train, y_train)

        # Predict on test set
        y_pred = pipe.predict(X_test)

        # Get accuracy, precision, recall, f1-score
        report = classification_report(y_test,y_pred)
        print("Classification report\n", report)

        results.append([model, {
            'model': name,
            'predictions': y_pred,
            'report': report,
        }])           

    return results



#  Visualize Results

def plot_distribution():
    """Distribution of posts

    Examine the number of samples collected from each forum
    """
    _, counts = np.unique(labels, return_counts=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8,3), dpi=120)
    plt.title("Number of posts by forum")
    sns.barplot(x=forums, y=counts)
    plt.legend([' '.join([f.title(),f"- {c} posts"]) 
                for f,c in zip(forums, counts)])
    plt.savefig('images/distribution.png')
    plt.show()

def plot_confusion(result):
    """
    Confusion Matrices
    """
    print("Classification report\n",
          result[-1]['report'])
    y_pred = result[-1]['predictions']
    conf_matrix = confusion_matrix(y_test,y_pred)
    _, test_counts = np.unique(y_test, return_counts=True)
    conf_matrix_percent =        conf_matrix / test_counts.transpose() * 100
    plt.figure(figsize=(9,8), dpi=120)
    plt.title(result[-1]['model'].upper() + " Results")
    plt.xlabel("Ground truth")
    plt.ylabel("Model prediction")
    ticklabels = [f"r/{sub}" for sub in forums]
    sns.heatmap(data=conf_matrix_percent, 
                xticklabels = ticklabels,
                yticklabels = ticklabels,
                annot=True, fmt='.2f')
    name = result[-1]['model']
    plt.savefig(f'images/{name}')
    plt.show()




if __name__ == "__main__":

    data, labels = load_data()
    X_train, X_test, y_train, y_test = split_data()
    pipeline = preprocessing_pipeline()
    all_models = load_models()
    
    # Run training/inference for all models
    results = fit_and_evaluate(all_models, pipeline,
                                X_train, X_test, 
                                y_train, y_test)

    print("Successfully completed training/inference for all models")

    # Plot distribution
    plot_distribution()

    for result in results:


        # Plot confusion matrix
        plot_confusion(result)
