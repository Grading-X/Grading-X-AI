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

# 네이버 쇼핑 리뷰 데이터
# urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")

data = pd.read_table('ratings_total.txt', names=['ratings', 'reviews'])
print(len(data))