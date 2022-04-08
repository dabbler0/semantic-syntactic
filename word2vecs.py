from tools import *
from gensim.models import Word2Vec
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from itertools import chain
import unidecode
nltk_tokenizer = TreebankWordTokenizer()
nltk_detokenizer = TreebankWordDetokenizer()

sentence_prop = InjectedProperty('sentence')

def word_tokenize(s):
    return nltk_tokenizer.tokenize(unidecode.unidecode(s))

def tokenize_dataset(d):
    return [
        word_tokenize(s) for s in d[sentence_prop]
    ]

word_token_prop = Property(
    [sentence_prop],
    'word_tokens',
    tokenize_dataset
)

def compute_vocab(d):
    return set(chain(*d[word_token_prop]))

vocab_prop = Property(
    [word_token_prop],
    'vocab',
    compute_vocab
)

def compute_word2vec(d):
    return Word2Vec(d[word_token_prop], vector_size=1024, min_count=1)

word2vec_prop = Property(
    [word_token_prop],
    'word2vec',
    compute_word2vec
)

# Compute it
if __name__ == '__main__':
    large = Dataset('large-0')

    word2vec_model = large[word2vec_prop]
