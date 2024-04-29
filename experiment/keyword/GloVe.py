import pandas as pd
from konlpy.tag import Okt
from glove import Corpus, Glove

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

corpus = Corpus()

corpus.fit(tokenized_data, window=5)
glove = Glove(no_components=100, learning_rate=0.05)

glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

glove.save('./save/glove/glove_model.model')
load_glove = Glove.load('./save/glove/glove_model.model')

similar_list = load_glove.most_similar("한국")
print(similar_list)