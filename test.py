import psycopg2
import Gpt
from sentence_transformers import SentenceTransformer, util
import fasttext
from konlpy.tag import Okt
import hgtk

sentence_model = SentenceTransformer('experiment/similarity/save/roberta-large_NLI_STS')
keyword_model = fasttext.load_model("./experiment/keyword/save/fasttext/fasttext.bin")
okt = Okt()

def grade(exam_content_id, grade_type):
    try:
        question_dic, question_guest_answer_dic, guest_answer_dic = fetch_queries_from_database(exam_content_id)

        final_score_dic = {}
        gpt_request_list = []

        for question_id in question_guest_answer_dic.keys():
            query = question_dic[question_id][0]
            correct_answer = question_dic[question_id][1]
            keyword_list = str(question_dic[question_id][2]).split('*')
            weightage = question_dic[question_id][3]

            answer_list = [correct_answer] + [guest_answer_dic[ga_id] for ga_id in question_guest_answer_dic[question_id]]

            embedding_list = sentence_model.encode(answer_list, batch_size=len(answer_list))
            cos_score_list = util.pytorch_cos_sim(embedding_list[0], embedding_list[1:])[0]

            #키워드 사전 처리
            keyword_pos_emb_flag_list = []
            if grade_type:
                for keyword in keyword_list:
                    word, pos = okt.pos(keyword)[0]
                    keyword_pos_emb_flag_list.append((word, pos, keyword_model[token_decompose(keyword)], False))

            for index in range(len(cos_score_list)):
                ga_id = question_guest_answer_dic[question_id][index]
                if cos_score_list[index] > 0.7:
                    final_score_dic[ga_id] = float(weightage)
                elif cos_score_list[index] < 0.4:
                    final_score_dic[ga_id] = float(0)
                else:
                    if grade_type:
                        for sublist in keyword_pos_emb_flag_list: sublist[3] = False
                        word_list = okt.pos(answer_list[index + 1])

                        for i, sublist in enumerate(keyword_pos_emb_flag_list):
                            keyword, key_pos, emb, flag = sublist
                            if flag: continue

                            for word, pos in word_list:
                                if pos == key_pos:
                                    score = util.pytorch_cos_sim(emb, keyword_model[token_decompose(word)]).item()
                                    if score > 0.57:
                                        keyword_pos_emb_flag_list[i][3] = True

                        ratio = sum(1 for item in keyword_pos_emb_flag_list if item[-1]) / len(keyword_pos_emb_flag_list)
                        keyword_score = ratio * 0.25
                        total_score = assign_score(cos_score_list[index]) + keyword_score
                        final_score_dic[ga_id] = float(total_score * weightage)
                    else:
                        tuple = (ga_id, query, correct_answer, guest_answer_dic[ga_id], weightage)
                        gpt_request_list.append(tuple)

        if not grade_type:
            gpt_response_dic = Gpt.main(gpt_request_list)
            final_score_dic.update(gpt_response_dic)

        return final_score_dic
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
    if 0.4 <= score < 0.5:
        return 0
    elif 0.5 <= score < 0.6:
        return 0.25
    elif 0.6 <= score < 0.65:
        return 0.5
    elif 0.65 <= score < 0.7:
        return 0.75

if __name__ == '__main__':
    exam_content_id = 1
    grade_type = False

    dic1, dic2, dic3 = fetch_queries_from_database(exam_content_id)
    print(dic1)
    print(dic2)
    print(dic3)

    final_dic = grade(exam_content_id, grade_type)
    print(final_dic)
