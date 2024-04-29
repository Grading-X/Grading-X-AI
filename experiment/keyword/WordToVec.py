import pandas as pd
import urllib.request
import gensim
from konlpy.tag import Okt

# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

train = pd.read_table("./save/word2vec/ratings.txt")
train = train.dropna(how='any')
print(train.isnull().values.any())
print("Total: ", len(train))

train['document'] = train['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()

tokenized_data = []
for sentence in train['document']:
    tokenized_sentence = okt.morphs(sentence, stem=True)
    no_stopwords_sentence = [word for word in tokenized_sentence if not word in stopwords]
    tokenized_data.append(no_stopwords_sentence)

model = gensim.models.Word2Vec(sentences = tokenized_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
print(model.wv.vectors.shape)

model.wv.save_word2vec_format('./save/word2vec/ko_w2v')
model = gensim.models.KeyedVectors.load_word2vec_format("./save/word2vec/ko_w2v")