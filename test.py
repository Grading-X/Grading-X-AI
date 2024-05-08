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

    '''
    GPU가 아닌 CPU 환경에서도 batch_size에 따라 속도 향상이 있음을 확인함 
    답안의 길이에 따라 소요 시간이 달라짐
    '''
    sentence_model.encode("Warmup")

    start = time.time()
    sentence_model.encode(answer_list, batch_size=1)
    end = time.time()
    print("Sequential Processing Time : ", end - start)
    print("모델이 할당된 디바이스:", sentence_model.device)

    print()

    start = time.time()
    sentence_model.encode(answer_list, batch_size=len(answer_list))
    end = time.time()
    print("Parallel Processing Time : ", end - start)
    print("모델이 할당된 디바이스:", sentence_model.device)

    embedding_list = sentence_model.encode(answer_list, batch_size=len(answer_list))

    student_answer = embedding_list[1:]
    print(student_answer.shape)
                                     # 1x1024   @   30x1024 transpose -> 1x30
    cos_score = util.pytorch_cos_sim(embedding_list[0], student_answer)[0] # index == 답안순서
    print(cos_score.shape)
    print(cos_score[0])


