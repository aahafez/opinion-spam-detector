
# Opinion Spam Detector

This project focuses on designing and implementing a system for detecting opinion spam—fraudulent or deceptive reviews—using machine learning models. The project emphasizes the application of Explainable AI (XAI) techniques, specifically the LIME (Local Interpretable Model-agnostic Explanations) library, to provide insights into the decision-making process of the model. Students will develop a model capable of classifying reviews as genuine or spam and use LIME to interpret and understand the model’s predictions.



## Documentation

### Loading Data

```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv("deceptive-opinion.csv")
df.head()
```

### Data Set Distribution

```python
df['deceptive'].value_counts().plot.bar()
```

### Text Vectorization:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df['text']).toarray()
features.shape
X=pd.DataFrame(data=features)
Y=df['deceptive'].astype(str)
df['deceptive']=df['deceptive'].astype(str)
```

### Train Naive Bayes Classifier

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)
clf = MultinomialNB().fit(X_train, y_train)
```

### Make Predictions

```python
y_pred=clf.predict(X_test)
```

### Classification Metrics

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

### Build Explainable Model Using Lime

```python
import lime
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['deceptive', 'truthful'])

# Pick an example to explain
idx = 405
example_text = df['text'].iloc[idx]

def predict_proba(texts):
    texts_transformed = tfidf.transform(texts).toarray()
    return clf.predict_proba(texts_transformed)

for idx in [100, 415, 405]:  # Change the indices as needed
    example_text = df['text'].iloc[idx]
    # Generate explanation for the chosen example
    exp = explainer.explain_instance(example_text, predict_proba, num_features=10)

    exp.show_in_notebook(text=example_text)
```
## Lime Explanation Examples

![Opinion 1](https://i.imgur.com/NocZC00.png)
![Opinion 2](https://i.imgur.com/hWD1oLb.png)


## Authors

- [Abdullah Abdelhafez](https://www.github.com/aahafez)
- [Farah Afifi](https://github.com/Farahafifii)

