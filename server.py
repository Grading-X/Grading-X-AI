import grpc
import grader_pb2
import grader_pb2_grpc
import psycopg2
import Gpt
from sentence_transformers import SentenceTransformer, util
import fasttext
from konlpy.tag import Okt
import hgtk
from concurrent import futures


class GraderServicer(grader_pb2_grpc.GraderServicer):
    def __init__(self):
        self.sentence_model = SentenceTransformer('experiment/similarity/save/roberta-large_NLI_STS')
        self.keyword_model = fasttext.load_model("./experiment/keyword/save/fasttext/fasttext.bin")
        self.okt = Okt()

    def grade(self, request, context):
        try:
            exam_content_id = request.exam_content_id
            grade_type = request.grade_type

            question_dic, question_guest_answer_dic, guest_answer_dic = fetch_queries_from_database(exam_content_id)

            final_score_dic = {}
            gpt_request_list = []

            for question_id in question_guest_answer_dic.keys():
                query = question_dic[question_id][0]
                correct_answer = question_dic[question_id][1]
                keyword_list = str(question_dic[question_id][2]).split('*')
                weightage = question_dic[question_id][3]

                answer_list = [correct_answer] + [guest_answer_dic[ga_id] for ga_id in question_guest_answer_dic[question_id]]

                embedding_list = self.sentence_model.encode(answer_list, batch_size=len(answer_list))
                cos_score_list = util.pytorch_cos_sim(embedding_list[0], embedding_list[1:])[0]

                #키워드 사전 처리
                keyword_pos_emb_flag_list = []
                if grade_type:
                    for keyword in keyword_list:
                        word, pos = self.okt.pos(keyword)[0]
                        keyword_pos_emb_flag_list.append((word, pos, self.keyword_model[token_decompose(keyword)], False))

                for index in range(len(cos_score_list)):
                    ga_id = question_guest_answer_dic[question_id][index]
                    if cos_score_list[index] > 0.7:
                        final_score_dic[ga_id] = float(weightage)
                    elif cos_score_list[index] < 0.5:
                        final_score_dic[ga_id] = float(0)
                    else:
                        if grade_type:
                            for sublist in keyword_pos_emb_flag_list: sublist[3] = False
                            word_list = self.okt.pos(answer_list[index + 1])

                            for i, sublist in enumerate(keyword_pos_emb_flag_list):
                                keyword, key_pos, emb, flag = sublist
                                if flag: continue

                                for word, pos in word_list:
                                    if pos == key_pos:
                                        score = util.pytorch_cos_sim(emb, self.keyword_model[token_decompose(word)]).item()
                                        if score > 0.57:
                                            keyword_pos_emb_flag_list[i][3] = True

                            ratio = sum(1 for item in keyword_pos_emb_flag_list if item[-1]) / len(keyword_pos_emb_flag_list)
                            keyword_score = ratio * 0.25
                            total_score = assign_score(cos_score_list[index]) + keyword_score
                            final_score_dic[ga_id] = float(total_score * weightage)
                        else:
                            tuple = (ga_id, query, correct_answer, weightage)
                            gpt_request_list.append(tuple)

            if not grade_type:
                gpt_response_dic = Gpt.main(gpt_request_list)
                final_score_dic.update(gpt_response_dic)

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
        host='localhost',
        port=5432
    )
    cur = conn.cursor()

    cur.execute("SELECT q.question_id, q.query, q.answer, q.keyword_list, q.weightage "
                "FROM question AS q "
                "WHERE q.exam_content_id = %s ",
                (exam_content_id,))

    question_dic = {}
    for row in cur.fetchall():
        question_dic[row[0]] = [row[1], row[2], row[3], row[4]]

    cur.execute("SELECT ga.question_id, ga.guest_answer_id, ga.answer "
                "FROM guest_answer AS ga "
                "WHERE ga.question_id IN ("
                    "SELECT q.question_id "
                    "FROM question AS q "
                    "WHERE q.exam_content_id = %s "
                ")", (exam_content_id,))


    question_guest_answer_dic = {}
    guest_answer_dic = {}
    for row in cur.fetchall():
        if row[0] not in question_guest_answer_dic:
            question_guest_answer_dic[row[0]] = []
        question_guest_answer_dic[row[0]].append(row[1])
        guest_answer_dic[row[1]] = row[2]


    cur.close()
    conn.close()

    return question_dic, question_guest_answer_dic, guest_answer_dic

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

def assign_score(score):
    if 0.5 <= score < 0.55:
        return 0
    elif 0.55 <= score < 0.6:
        return 0.25
    elif 0.6 <= score < 0.65:
        return 0.5
    elif 0.65 <= score < 0.7:
        return 0.75

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grader_pb2_grpc.add_GraderServicer_to_server(GraderServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
