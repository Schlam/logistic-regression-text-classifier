from nltk.corpus import stopwords
import re

# Remove urls using "The perfect regex URL pattern"
def preprocessor(doc):

    pattern = (r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    
    return re.sub(pattern, " ", doc)

    
# List of subreddits to use as a basis for our model
forums = [
    'astrology',
    'datascience',
    'machinelearning',
    'conspiracy',
    'physics'
]

# Parameters used to get partition train/test data
TEST_SIZE = .2
RANDOM_STATE = 0


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
# TfidfVectorizer
ANALYZER = 'word'
MIN_DOC_FREQ = 2
STOP_WORDS = set(stopwords.words('english'))
NGRAM_RANGE = (1,3)

# Singular Value Decomposition
N_PRINCIPAL_COMPONENTS = 1024
N_ITER = 16



"""Set the following model parameters

    For 'RandomForestClassifier'
        Currently using default parameters

    For 'LogisticRegressionCV'
        CV: number of cross-validations to make when fitting
"""

# RandomForestClassifier
N_JOBS = 1

# LogisticRegressionCV
REGULARIZATION_STRENGTH = 5
CROSS_VALIDATION_FOLDS = 5
MAX_ITER = 120
LOGISTIC_REGRESSION_SOLVER = 'saga'


partition_params = {
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
}

embedding_params = [
    {
        "preprocessor": preprocessor,
        "analyzer": ANALYZER,
        "ngram_range": NGRAM_RANGE,
        'stop_words': STOP_WORDS,
        'min_df': MIN_DOC_FREQ,
    },
    {
        'n_components': N_PRINCIPAL_COMPONENTS,
        'n_iter': N_ITER,
    }
]

model_params = [
    {
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
    },{
        "Cs": REGULARIZATION_STRENGTH,
        "cv": CROSS_VALIDATION_FOLDS, 
        "random_state": RANDOM_STATE,
        "max_iter": MAX_ITER,
        "solver": LOGISTIC_REGRESSION_SOLVER,
        "n_jobs": N_JOBS,
    }
]