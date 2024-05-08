import grpc
import grader_pb2
import grader_pb2_grpc
import psycopg2
from sentence_transformers import SentenceTransformer, util
from concurrent import futures


class GraderServicer(grader_pb2_grpc.GraderServicer):
    def __init__(self):
        self.model = SentenceTransformer('experiment/similarity/save/roberta-large_NLI_STS')

    def grade(self, request, context):
        try:
            exam_content_id = request.exam_content_id
            guest_email = request.guest_email

            answer_dic = fetch_queries_from_database(exam_content_id, guest_email)

            cos_score_dic = {}

            print(answer_dic)

            print("Received request for exam_content_id:", exam_content_id)
            print("Received request for guest_email:", guest_email)
            print("Answer dictionary:", answer_dic)

            for question_id, value in answer_dic.items():
                correct_answer = value[0].split('*')
                guest_answer = value[1]
                max_score = 0
                for correct in correct_answer:
                    embedding = self.model.encode([correct, guest_answer])
                    cos_score = util.pytorch_cos_sim(embedding[0], embedding[1])[0]
                    max_score = max(max_score, cos_score.item())
                cos_score_dic[question_id] = max_score

            return grader_pb2.GradingResponse(cosine_similarity=cos_score_dic)
        except Exception as e:
            # 오류 출력
            print("Error occurred during grading:", e)
            raise


def fetch_queries_from_database(exam_content_id, guest_email):
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='postgres',
        host='3.34.49.173',
        port=5432
    )
    cur = conn.cursor()

    cur.execute("SELECT q.question_id, q.answer_list, ga.answer "
                "FROM question AS q "
                "LEFT JOIN guest_answer AS ga ON q.question_id = ga.question_id "
                "WHERE q.exam_content_id = %s "
                "AND ga.guest_email = %s "
                "ORDER BY q.index ASC",
                (exam_content_id, guest_email))

    answer_dic = {}
    for row in cur.fetchall():
        answer_dic[row[0]] = [row[1], row[2]]

    cur.close()
    conn.close()

    return answer_dic


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grader_pb2_grpc.add_GraderServicer_to_server(GraderServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
