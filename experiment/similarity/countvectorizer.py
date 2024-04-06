import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 별도의 데이터 파일 필요
train = pd.read_csv('./train.csv')
test = pd.read_csv("./test.csv")

class SimpleModel():
  def __init__(self, threshold=0.5):
    super(SimpleModel, self).__init__()
    self.threshold = threshold
    self.vectorizer = CountVectorizer()

  def fit(self, sentence1, sentence2):
    self.vectorizer.fit(sentence1)
    self.vectorizer.fit(sentence2)

  def predict(self, sentence1, sentence2):
    vectors1 = self.vectorizer.transform(sentence1)
    vectors2 = self.vectorizer.transform(sentence2)

    prediction = []
    for vector1, vector2 in zip(vectors1, vectors2):
        prediction.append(cosine_similarity(vector1, vector2))
    prediction = np.reshape(prediction, len(prediction))
    prediction = np.where(prediction > self.threshold, 1, 0)

    return prediction

# Model fitting
model = SimpleModel(threshold=0.5)
model.fit(train['sentence1'], train['sentence2'])

# Model Inference
preds = model.predict(test['sentence1'], test['sentence2'])