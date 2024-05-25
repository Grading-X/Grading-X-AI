import os

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

if __name__ == '__main__':
    file_path = "./data/example2.txt" # pdf나 txt파일 예상하고 작업진행했습니다. 어떤식으로 주셔야할듯합니다??

    extension = get_file_extension(file_path) # 파일 확장자 확인