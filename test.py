import numpy as np
import time
from sentence_transformers import SentenceTransformer, util
import fasttext
from konlpy.tag import Okt


if __name__ == '__main__':
    sentence_model = SentenceTransformer('experiment/similarity/save/roberta-large_NLI_STS')
    keyword_model = fasttext.load_model("./experiment/keyword/save/fasttext/fasttext.bin")
    okt = Okt()




