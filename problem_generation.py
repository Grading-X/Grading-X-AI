import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from pipeline import pipeline_prompt
from dotenv import load_dotenv
load_dotenv()

def is_pdf(file_path):
    with open(file_path, 'rb') as file:
        header = file.read(4)
        return header == b'%PDF'

def is_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file.read()
        return True
    except UnicodeDecodeError:
        return False

def get_file_extension(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return 'PDF'
    elif file_extension.lower() == '.txt':
        return 'TXT'
    else:
        if is_pdf(file_path):
            return 'PDF'
        elif is_txt(file_path):
            return 'TXT'
        else:
            return 'Unknown'

def processing_pdf(path):
    loader = PyPDFLoader(path)
    document = loader.load()

    context = ''
    for i in range(len(document)):
        context += document[i].page_content + "\n"

    return context

def processing_txt(path):
    with open(path, 'r', encoding='utf-8') as file:
        context = file.read()
    return context

def problem_generation(N):
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o",
                     openai_api_key=os.environ.get("OPENAI_API_KEY"))

    formatted_prompt = pipeline_prompt.format(
        context=context,
        example_q1="형식용 문장",
        example_a1="형식용 문장",
        example_q2="형식용 문장",
        example_a2="형식용 문장",
        example_q3="형식용 문장",
        example_a3="형식용 문장",
        N=N,
    )

    sentence = llm.invoke(formatted_prompt).content
    return sentence

def extract_qa_pairs(text):
    qa_pattern = re.compile(r'문제:\s*(.*?)\s*답안:\s*(.*?)(?=\s*문제:|\s*$)', re.DOTALL)
    matches = qa_pattern.findall(text)

    qa_pairs = []

    for match in matches:
        question, answer = match
        qa_pairs.append({'문제': question.strip(), '답안': answer.strip()})

    return qa_pairs

if __name__ == '__main__':
    file_path = "./data/example2.txt" # pdf나 txt파일 예상하고 작업진행했습니다. 어떤식으로 주셔야할듯합니다??

    extension = get_file_extension(file_path) # 파일 확장자 확인

    # PDF인 경우 처리
    if extension == 'PDF':
        context = processing_pdf(file_path)  # 파일에서 정보를 뽑아냅니다
        sentence = problem_generation(N=1)  # 문제를 N개 생성합니다 (너무 많이 X..., 1개 권장)

        qa_pairs = extract_qa_pairs(sentence)  # 문제, 답안을 key로 dictionary로 구성합니다
        for pair in qa_pairs:  # 지정한 문제 수 N개만큼 dictionary로 구성되어 있습니다.
            print(f"문제: {pair['문제']}")
            print(f"답안: {pair['답안']}")

    # TXT인 경우 처리
    elif extension == 'TXT':
        context = processing_txt(file_path)  # 파일에서 정보를 뽑아냅니다
        sentence = problem_generation(N=1)  # 문제를 N개 생성합니다 (너무 많이 X..., 1개 권장)

        qa_pairs = extract_qa_pairs(sentence)  # 문제, 답안을 key로 dictionary로 구성합니다
        for pair in qa_pairs:  # 지정한 문제 수 N개만큼 dictionary로 구성되어 있습니다.
            print(f"문제: {pair['문제']}")
            print(f"답안: {pair['답안']}")

    else:
        # 처리할 수 없는 파일 확장자인 경우입니다. -> 에러처리? 프론트/백에서 검증거치고 들어오는지?
        pass