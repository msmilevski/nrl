import re
from pymystem3 import Mystem
import sys
import pandas as pd
from transliterate import translit
from tqdm import tqdm

def preprocess_line(line, reg, mystem=None):

    line = line.lower()
    line = reg.sub("", line)
    line = re.sub('\d+', '0', line)
    line = re.sub('\n', '', line)
    line = translit(line, 'ru')

    if(mystem != None):
        line = mystem.lemmatize(line)
        line = "".join(line)

    line = line + '\n'

    return line


data_path = sys.argv[1]
processed_text_file_path = sys.argv[2]
reader = pd.read_csv(data_path, chunksize=10g0, encoding='utf-8')
processed_text_file = open(processed_text_file_path, 'a')
reg = re.compile('[^a-z^A-Z^0-9^А-я^\s*]')
lemmatiziation = sys.argv[3]


if lemmatiziation:
    mystem = Mystem()

for batch in tqdm(reader):
    descriptions = batch['description']
    temp_text = ""
    for desc in descriptions:
        if lemmatiziation:
            temp_text = temp_text + preprocess_line(desc, reg, mystem)
        else:
            temp_text = temp_text + preprocess_line(desc, reg)

    processed_text_file.write(temp_text)