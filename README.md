# Training a semantic classifier with Reddit data


The following code pulls the 1000 newest posts from each designated subreddit,
embeds each post using Tf-idf, and predicts the forum using logistic regression.

### Reproducible

All data and parameters used to create the model are loaded into mongoDB.
The entire pipeline is also saves as a pickle file. ```TFIDF_SVD_LRCV.pkl```

The pipeline:

```

# Text vectorizer
vector = TfidfVectorizer(min_df=2,
                         max_features=3200,
                         preprocessor=preprocessor)


# Principal component analyzer
decomp = TruncatedSVD(n_iter=8,
                      random_state=0,
                      n_components=1000)


# Regression model
model = LogisticRegressionCV(Cs=8, 
                             cv=5, 
                             n_jobs=-1,
                             max_iter=200,
                             solver='saga',
                             random_state=0)

```

## Results

### Classification report

```
                 precision    recall  f1-score   support

machinelearning       0.72      0.87      0.79        78
    datascience       0.74      0.70      0.72        64
     conspiracy       0.88      0.85      0.86        41
      astrology       0.94      0.92      0.93        53
      chemistry       0.78      0.79      0.79        72
       starwars       0.79      0.87      0.82        38
        biology       0.58      0.62      0.60        50
        physics       1.00      0.42      0.59        12
         gaming       0.96      0.69      0.80        32
          vegan       0.88      0.79      0.84        29

       accuracy                           0.78       469
      macro avg       0.83      0.75      0.77       469
   weighted avg       0.80      0.78      0.78       469

```

### Confusion Matrix

![C](images/confusion.png)
