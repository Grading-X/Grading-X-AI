import os
from langchain_community.document_loaders import PyPDFLoader

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

if __name__ == '__main__':
    file_path = "./data/example2.txt" # pdf나 txt파일 예상하고 작업진행했습니다. 어떤식으로 주셔야할듯합니다??

    extension = get_file_extension(file_path) # 파일 확장자 확인

    # PDF인 경우 처리
    if extension == 'PDF':
        context = processing_pdf(file_path)  # 파일에서 정보를 뽑아냅니다
        pass

    # TXT인 경우 처리
    elif extension == 'TXT':
        pass

    else:
        # 처리할 수 없는 파일 확장자인 경우입니다. -> 에러처리? 프론트/백에서 검증거치고 들어오는지?
        pass