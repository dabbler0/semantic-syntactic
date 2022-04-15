from tools import *
import nltk
from nltk import tokenize
import datasets
import os
from tqdm import tqdm

scratch_dir = '/raid/lingo/abau/scratch/'

french_file = os.path.join(scratch_dir, 'news.2021.fr.shuffled.deduped')
chinese_file = os.path.join(scratch_dir, 'news.2021.zh.shuffled.deduped')

def create_dataset_of_size(raw_file, dataset_name, size = 25163, offset = 0):
    cursor = 0
    result = []
    with open(raw_file, 'r') as raw_lines:
        for line in tqdm(raw_lines):
            if cursor >= offset:
                result.append(line)
            if len(result) >= size:
                break
            cursor += 1

        create_dataset(dataset_name, {
            'sentence': result
        })

for n in range(3):
    create_dataset_of_size(
        french_file,
        'fr-tiny-%d',
        offset = 25163 * n
    )

    create_dataset_of_size(
        chinese_file,
        'zh-tiny-%d',
        offset = 25163 * n
    )
