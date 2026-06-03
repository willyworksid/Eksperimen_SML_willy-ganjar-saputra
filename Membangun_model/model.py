import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

df = pd.read_csv("./preprocessing/dataset_preprocessing.csv")

print(df.head())
print(df.columns)
print(df.shape)

print(df['sentiment'].value_counts())

df['text_final'].isnull().sum()

cv = CountVectorizer()

x_count = cv.fit_transform(df['text_final'])

freq = (x_count > 0).sum(axis=0).A1

print("Kata muncul 1 dokumen :", (freq == 1).sum())
print("Kata muncul 2 dokumen :", (freq == 2).sum())
print("Kata muncul 3 dokumen :", (freq == 3).sum())
print("Total vocabulary :", len(freq))


x = df['text_final']
y = df['sentiment']

print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train :", x_train.shape)
print("Test  :", x_test.shape)

mlflow.set_tracking_uri(
    'http://localhost:5000'
)

mlflow.set_experiment(
    'Sentiment_Analysis_DANA_willy-ganjar-saputra'
)

print(mlflow.get_tracking_uri())

mlflow.sklearn.autolog()

y_train = y_train.values
y_test = y_test.values

with mlflow.start_run():

  model = Pipeline([
    (
        'tfidf',
        TfidfVectorizer(
            max_features=5000,
            min_df=2
        )
    ),
    (
        'logreg',
        LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
    )
  ])

  model.fit(x_train,y_train)

  y_pred = model.predict(x_test)

  accuracy = accuracy_score(
    y_test,
    y_pred
  )
  
  print(
    "Accuracy:",
    accuracy
  )

  print(
    classification_report(
        y_test,
        y_pred
    )
  )

  mlflow.sklearn.log_model(
        sk_model=model,
        name="model"
    )

  print("Model berhasil disimpan ke MLflow")