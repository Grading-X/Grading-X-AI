'''
Colab Environment Setting

!pip install konlpy
!pip install hgtk
!git clone https://github.com/facebookresearch/fastText.git
%cd fastText
!make
!pip install .
'''

import urllib.request
import pandas as pd
import hgtk
from konlpy.tag import Okt
import fasttext

# 네이버 쇼핑 리뷰 데이터
# urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")

data = pd.read_table('ratings_total.txt', names=['ratings', 'reviews'])
print(len(data))

def token_decompose(token):
    def special_token(consonant):
        if consonant: return consonant
        return '-'

    decomposed_token = ''
    for char in token:
        try:
            initial, neutral, final = hgtk.letter.decompose(char)

            initial = special_token(initial)
            neutral = special_token(neutral)
            final = special_token(final)
            decomposed_token = decomposed_token + initial + neutral + final
        except Exception as exception:
            if type(exception).__name__ == 'NotHangulException':
                decomposed_token += char

    return decomposed_token

print(token_decompose("테스트용"))

okt = Okt()

def morpheme_tokenize(sentence):
    return [token_decompose(token) for token in okt.morphs(sentence)]

# 전처리 약 20분 소요 -> 저장해두고 사용할 것
train = []
for i, sentence in enumerate(data['reviews'].to_list()):
    preprocessed_data = morpheme_tokenize(sentence)
    train.append(preprocessed_data)

    if i % 10000 == 0: print(f"{i} Done")

with open("train.txt", 'w') as f:
    for line in train:
        f.write(' '.join(line)+'\n')

model = fasttext.train_unsupervised('train.txt', model='cbow')
model.save_model("fasttext.bin")
model = fasttext.load_model("fasttext.bin")

print(model.get_nearest_neighbors(token_decompose('형태'), k=5))

def assemble_to_word(sequence):
    char_splitted = []
    index = 0

    while index < len(sequence):
        if not hgtk.checker.is_hangul(sequence[index]):
            char_splitted.append(sequence[index])
            index = index + 1
        else:
            char_splitted.append(sequence[index:index+3])
            index = index + 3

    word = ''
    try:
        for char in char_splitted:
            if len(char) == 3:
                if char[2] == "-":
                    word = word + hgtk.letter.compose(char[0], char[1])
                else:
                    word = word + hgtk.letter.compose(char[0], char[1], char[2])
            else:
                word = word + char
    except Exception as exception:
        if type(exception).__name__ == 'NotHangulException':
            return sequence

    return word

def pipeline(sequence):
    return [(assemble_to_word(word), similarity) for (similarity, word) in sequence]

print(pipeline(model.get_nearest_neighbors(token_decompose('물티슈'), k=5)))