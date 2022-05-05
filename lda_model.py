from tools import *
from languages import *
from tqdm import tqdm, trange
import numpy as np
import lda

sentence_prop = InjectedProperty('sentence')
def get_wordtokens(d, language=ENGLISH_NLTK):
    return [
        tuple(language.tokenize(sentence))
        for sentence in tqdm(d[sentence_prop])
    ]

wordtoken_prop = Property(
    [sentence_prop],
    'wordtokens2',
    get_wordtokens
)

def get_wordvocab(d, size=20000):
    counts = {}
    for sentence in tqdm(d[wordtoken_prop]):
        for tok in sentence:
            if tok not in counts:
                counts[tok] = 0
            counts[tok] += 1

    result = sorted(counts, key = lambda x: counts[x])[:size]
    inverse_result = { t: i for i, t in enumerate(result) }
    return result, inverse_result

wordvocab_prop = Property(
    [wordtoken_prop],
    'wordvocab2',
    get_wordvocab
)

def get_numerical_wordtokens(d):
    vocab, lookup = d[wordvocab_prop]
    return [
        [lookup.get(x, len(vocab) + 1) for x in sentence]
        for sentence in tqdm(d[wordtoken_prop])
    ]

nw_prop = Property(
    [wordtoken_prop, wordvocab_prop],
    'numerical_wordtokens2',
    get_numerical_wordtokens
)

def train_lda(d, n = 100, vocab_size = 20000, language=ENGLISH_NLTK):
    X = np.array(d[nw_prop])

    model = lda.LDA(n_topics=n, n_iter=1500, random_state=1)
    model.fit(X)

    return model.components_

lda_prop = Property(
    [nw_prop],
    'lda_params',
    train_lda
)

if __name__ == '__main__':
    tiny = Dataset('tiny-2')

    lda_model = tiny[lda_prop]

