## Training a semantic classifier with Reddit data

The following code pulls the 1000 newest posts from each designated subreddit, vectorizes each post and uses logistic regression to predict the subreddit of origin.

### Update (12/20)

Added mongoDB functionality to save both the text data and hyperparameters for our model/pipeline. The trained model is saves as  ```TFIDF_SVD_LRCV.pkl``` and
is included in this repository
