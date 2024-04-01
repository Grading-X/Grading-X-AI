import time
from sentence_transformers import SentenceTransformer, util

if __name__ == '__main__':
    start = time.time()
    model = SentenceTransformer('experiment/save/roberta-large_NLI_STS') # 해당 경로에 있는 파일이 필요합니다. 다운받으셔서 경로 지정해주세요
    end = time.time()
    print("model load time:", end-start, "sec")

    # 모범 정답, 학생 정답에 대한 예시 데이터입니다.
    query_list = [["이진 탐색 시 배열에 저장된 데이터는 정렬되어 있어야 한다.", "이진 탐색을 수행할 때, 배열은 사전에 정렬되어 있어야함."],
                  ["마찰이나 공기저항이 없을 때 역학적 에너지는 항상 일정하게 보존된다.", "역학적 에너지는 항상 일정하게 보존된다."],
                  ["개개인의 생각을 자유롭게 표현할 수 있다.", "개개인의 의견은 탄압되고 제한된다."],
                  ["시맨틱 태그는 시맨틱 웹을 구현할 때 사용하는 방법이다.", "모름"]]

    for query in query_list:
        start = time.time()
        embedding = model.encode(query)
        cos_score = util.pytorch_cos_sim(embedding[0], embedding[1])[0]
        end = time.time()

        print("Elapsed Time: ", end-start, "Score: ", cos_score.item()) # 최종 정답 점수를 얻을 수 있습니다.

    '''
    cos_score 값에 대한 설명은 다음과 같습니다. 
    https://arxiv.org/pdf/2105.09680.pdf ( 17page 참조 ) 
    
    1: 중요한 정보, 중요하지 않은 정보가 동일하다. 
    0.8: 중요한 정보는 동일하나, 몇몇 중요하지 않은 정보가 다르다.
    0.6: 중요한 정보는 동일하나, 중요하지 않은 정보가 무시할 수 없는 수준이다.( 의미에 영향이 갈 수 있음 )
    0.4: 중요한 정보가 동일하지 않으며, 중요하지 않은 정보만을 공유하고 있다.
    0.2: 모든 정보가 유사하지 않으나 같은 주제만을 다루고 있다. 
    0: 완전히 동일하지 않다. 
    
    0.7 ~ 1 -> 정답처리 
    0.5 ~ 0.7 -> 추가 처리가 필요함... 
    0 ~ 0.5 -> 오답처리
    '''