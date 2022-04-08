from word2vecs import *
import numpy as np

large = Dataset('large-0')

vocab = list(large[vocab_prop])
wv = large[word2vec_prop].wv

def create_parallelogram(word):
    origin = wv[word]
    # Randomly select a couple of other keys
    key1 = np.random.choice(vocab)
    vec1 = wv[key1]

    key2 = np.random.choice(vocab)
    vec2 = wv[key2]

    dest = vec1 + vec2 - origin

    # Find nearest neighbor to dest within wv

    return word, key1, key2, wv.most_similar(positive=[key1, key2], negative=origin)

for _ in range(100):
    print(create_parallelogram('mayor'))
