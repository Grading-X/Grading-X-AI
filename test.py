import numpy as np
import time
from sentence_transformers import SentenceTransformer, util
import fasttext
from konlpy.tag import Okt

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

if __name__ == '__main__':
    sentence_model = SentenceTransformer('experiment/similarity/save/roberta-large_NLI_STS')
    keyword_model = fasttext.load_model("./experiment/keyword/save/fasttext/fasttext.bin")
    okt = Okt()

    # 테스트용 데이터
    answer_list = ['대한민국은 아시아의 국가 중 하나이다.']  # 0 index에 모범 정답 할당
    keyword_list = ['대한민국', '아시아']  # 초기 키워드
    answer_list.append('대한민국은 아시아 국가')
    answer_list.append('대한민국은 아프리카 국가')
    answer_list.append('일본은 아시아 국가')
    answer_list.append('일본은 아프리카 국가')

    embedding_list = sentence_model.encode(answer_list, batch_size=len(answer_list))
    student_answer = embedding_list[1:]
    print(student_answer.shape)
                                     # 1x1024   @   30x1024 transpose -> 1x30
    cos_score = util.pytorch_cos_sim(embedding_list[0], student_answer)[0] # index == 답안순서
    print(cos_score.shape)
    print(cos_score[0])


    final_score = [0] * (len(answer_list) - 1)  # 0~1점 사이로 mapping함, 추후 배점에 맞게 곱연산 필요
    keyword_index = []
    for index, score in enumerate(cos_score):
        if score >= 0.7:
            final_score[index] = 1
        elif score <= 0.5:
            final_score[index] = 0
        else:
            keyword_index.append(index)
