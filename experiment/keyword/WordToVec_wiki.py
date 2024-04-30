!pip install wikiextractor
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh

!wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
!python -m wikiextractor.WikiExtractor kowiki-latest-pages-articles.xml.bz2

import re
import os
from konlpy.tag import Mecab
from gensim.models import Word2Vec

def list_wiki(dirname):
    filepaths = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        filepath = os.path.join(dirname, filename)

        if os.path.isdir(filepath):
            filepaths.extend(list_wiki(filepath))
        else:
            find = re.findall(r"wiki_[0-9][0-9]", filepath)
            if 0 < len(find):
                filepaths.append(filepath)
    return sorted(filepaths)

filepaths = list_wiki("text")

with open("output_file.txt", "w") as outfile:
    for filename in filepaths:
        with open(filename) as infile:
            contents = infile.read()
            outfile.write(contents)

f = open('output_file.txt', encoding="utf8")

i = 0
while True:
    line = f.readline()
    if line != '\n':
        i = i+1
        print("%d번째 줄 :"%i + line)
    if i==10:
        break
f.close()

mecab = Mecab()

f = open('output_file.txt', encoding="utf8")
lines = f.read().splitlines()
print(len(lines))

result = []
for line in lines:
  if line:
    result.append(mecab.morphs(line))
print(len(result))

model = Word2Vec(result, size=100, window=5, min_count=5, workers=4, sg=0)