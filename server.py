import grpc
import grader_pb2
import grader_pb2_grpc
import psycopg2
from sentence_transformers import SentenceTransformer, util
import fasttext
from konlpy.tag import Okt
from concurrent import futures


class GraderServicer(grader_pb2_grpc.GraderServicer):
    def __init__(self):
        self.sentence_model = SentenceTransformer('experiment/similarity/save/roberta-large_NLI_STS')
        self.keyword_model = fasttext.load_model("./experiment/keyword/save/fasttext/fasttext.bin")
        self.okt = Okt()

    def grade(self, request, context):
        try:
            exam_content_id = request.exam_content_id

            answer_dic, guest_dic = fetch_queries_from_database(exam_content_id)

            final_score_dic = {}

            for question_id, guest_answer_list in guest_dic.items():
                correct_answer = answer_dic[question_id][0]
                keyword = answer_dic[question_id][1].split('*')
                weightage = answer_dic[question_id][2]
                answer_list = [correct_answer] + [ga[1] for ga in guest_answer_list]
                embedding_list = self.sentence_model.encode(answer_list, batch_size=len(answer_list))
                cos_score_list = util.pytorch_cos_sim(embedding_list[0], embedding_list[1:])[0]
                for index in range(len(guest_answer_list)):
                    final_score_dic[guest_answer_list[index][0]] = processScore(cos_score_list[index])

            return grader_pb2.GradingResponse(final_score=final_score_dic)
        except Exception as e:
            # 오류 출력
            print("Error occurred during grading:", e)
            raise


def fetch_queries_from_database(exam_content_id):
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='postgres',
        host='3.34.49.173',
        port=5432
    )
    cur = conn.cursor()

    #1. 시험에 해당하는 모든 문제 - 문제id, 키워드, 모범답안, 배점 가져옴
    cur.execute("SELECT q.question_id, q.answer, q.keyword_list, q.weightage "
                "FROM question AS q "
                "WHERE q.exam_content_id = %s ",
                exam_content_id)

    answer_dic = {}
    for row in cur.fetchall():
        answer_dic[row[0]] = [row[1], row[2], row[3]]

    #2. 시험에 해당하는 모든 GuestExamContent에서 guest email 가져옴
    #3. 2에서 가져온 guest_email 로 모든 GuestAnswer - 문제id, 답안

    cur.execute("SELECT ga.question_id, ga.guest_answer_id, ga.answer "
                "FROM quest_answer AS ga "
                "WHERE ga.guest_email IN ("
                    "SELECT gec.guest_email "
                    "FROM guest_exam_content AS gec "
                    "WHERE gec.exam_content_id = %s "
                ")", exam_content_id)


    guest_dic = {}
    for row in cur.fetchall():
        if row[0] not in guest_dic:
            guest_dic[row[0]] = []
        guest_dic[row[0]].append([row[1], row[2]])

    cur.close()
    conn.close()

    return answer_dic, guest_dic


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grader_pb2_grpc.add_GraderServicer_to_server(GraderServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
