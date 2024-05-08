import numpy as np
import time
from sentence_transformers import SentenceTransformer, util
import fasttext
from konlpy.tag import Okt


if __name__ == '__main__':
    sentence_model = SentenceTransformer('experiment/similarity/save/roberta-large_NLI_STS')
    keyword_model = fasttext.load_model("./experiment/keyword/save/fasttext/fasttext.bin")
    okt = Okt()

    # 병렬 처리 테스트용 데이터
    answer_list = ['안녕하세요']  # 0 index에 모범 정답 할당
    keyword = ['키워드1', '키워드2']  # 초기 키워드
    for char in ('안', '녕', '하', '세', '요') * 6:  # 30개 답안 구성
        answer_list.append(char)



