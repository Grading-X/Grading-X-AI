import numpy as np
import time
from sentence_transformers import SentenceTransformer, util
import fasttext
from konlpy.tag import Okt
import hgtk

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

    # keyword는 사전에 품사태깅, 임베딩 과정을 저장해놓습니다.
    keyword_pos_emb_flag_list = []
    for keyword in keyword_list:
        word, pos = okt.pos(keyword)[0]
        keyword_pos_emb_flag_list.append([word, pos, keyword_model[token_decompose(keyword)], False])

    threshold = 0.57
    final_score = [0] * (len(answer_list) - 1)  # 0~1점 사이로 mapping함, 추후 배점에 맞게 곱연산 필요
    for index, score in enumerate(cos_score):
        if score >= 0.7:
            final_score[index] = 1
        elif score <= 0.5:
            final_score[index] = 0
        else:
            # 평가할 문장을 형태소분석/품사태깅, 각 형태소에 대해 임베딩 비교합니다.
            word_list = okt.pos(
                answer_list[index + 1])  # index+1에 학생답안이 저장되어 있으므로 이렇게했습니다. answer_list 구성하는 방법에 따라 자유롭게 진행하시면됩니다.
            for i, sublist in enumerate(keyword_pos_emb_flag_list):
                keyword, key_pos, emb, flag = sublist
                if flag: continue

                for word, pos in word_list:  # 처리할 문장
                    if pos == key_pos:
                        if util.pytorch_cos_sim(emb, keyword_model[token_decompose(word)]).item() > threshold:
                            keyword_pos_emb_flag_list[i][3] = True  # flag -> True
