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