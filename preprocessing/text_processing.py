import re
from pymystem3 import Mystem
import sys
import pandas as pd
from transliterate import translit
from tqdm import tqdm


def preprocess_line(line, reg, mystem=None):
    line = line.lower()
    line = re.sub('\d+', '0 ', line)
    line = re.sub('\.', '. ', line)
    line = reg.sub("", line)
    line = re.sub('\s\s+', ' ', line)
    line = re.sub("\n", '', line)

    line = line.strip()
    line = translit(line, 'ru')

    if (mystem != None):
        line = mystem.lemmatize(line)
        line = "".join(line)

    #line = line + '\n'

    return line


data_path = sys.argv[1]
processed_text_file_path = sys.argv[2]
reader = pd.read_csv(data_path, chunksize=100, encoding='utf-8')
reg = re.compile('[^a-z^A-Z^0-9^А-я^\s*]')
lemm = (sys.argv[3] == 'True')

item_id = []
descriptions = []

if lemm == True:
    mystem = Mystem()

print(lemm)
for batch in tqdm(reader):
    descriptions = batch['description']
    ids = batch['itemID']
    temp_text = ""
    for i, desc in enumerate(descriptions):
        if lemm == True:
            descriptions.append(preprocess_line(desc, reg, mystem))
        else:
            descriptions.append(preprocess_line(desc, reg))
        item_id.append(ids[i])


d = {}
d['itemID'] = item_id
d['descriptions'] = descriptions
pd.DataFrame(data=d).to_csv(processed_text_file_path)