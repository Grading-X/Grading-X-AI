# Reference from https://konlpy.org/ko/v0.6.0/morph/

from konlpy.tag import Kkma, Komoran, Hannanum, Okt, Mecab

tagger_dict = {'Kkma':Kkma(),
               'Komoran': Komoran(),
               'hannanum': Hannanum(),
               'Okt': Okt()}
#Mecab = Mecab() # Window에선 지원되지 않음, 기타 환경에서 구동

sentence_list = ["아버지가방에들어가신다.",
                 "나는 밥을 먹는다",
                 "하늘을 나는 자동차",
                 "아이폰 기다리다 지쳐 애플공홈에서 언락폰질러버렸다 6+ 128기가실버ㅋ"] # OOV 처리 확인

keyword_list = [['아버지', '방'],
                ['밥'],
                ['하늘', '자동차'],
                ['아이폰', '애플']]

# 성능분석은 정성적 평가로 진행
for i, sentence in enumerate(sentence_list):
    print(f"--------------Sentence {i}--------------")

    for tagger in tagger_dict.keys():
        morpheme_list = tagger_dict[tagger].morphs(sentence)
        if all( [True if word in morpheme_list else False for word in keyword_list[i]] ):
            print("Keyword O,", end=' ')
        else:
            print("Keyword X,", end=' ')

        print(f"{tagger}: ", morpheme_list)
        print()
