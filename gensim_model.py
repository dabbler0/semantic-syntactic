from tools import *
import torch
from languages import *
from tqdm import tqdm, trange
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.stem.wordnet import WordNetLemmatizer
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

lemmatizer = WordNetLemmatizer()

def lemmatize(d):
    return [
        tuple(lemmatizer.lemmatize(t).lower() for t in sentence)
        for sentence in tqdm(d[wordtoken_prop])
    ]

lemmatized_prop = Property(
    [wordtoken_prop],
    'lemmatized_words3',
    lemmatize
)

def train_gensim(d):
    docs = d[lemmatized_prop] #wordtoken_prop]
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    temp = dictionary[0]
    id2word = dictionary.id2token
    chunksize = 2048
    num_topics = 128
    passes = 20
    iterations = 400

    model = LdaModel(
        corpus = corpus,
        id2word = id2word,
        chunksize = chunksize,
        alpha = 'auto',
        eta = 'auto',
        iterations = iterations,
        num_topics = num_topics,
        passes = passes,
        eval_every = None
    )

    return dictionary, model

gensim_prop = Property(
    [wordtoken_prop],
    'gensim_model_lemmatized_128topics',
    train_gensim
)

if __name__ == '__main__':
    tiny2 = Dataset('tiny-2')

    dictionary, model = tiny2[gensim_prop]

    def score_pipeline(sentence):
        bow = dictionary.doc2bow([lemmatizer.lemmatize(x).lower() for x in ENGLISH_NLTK.tokenize(sentence)])

        topics = torch.Tensor(model.get_topics())

        bow_tensor = torch.zeros_like(topics[0])
        for index, count in bow:
            bow_tensor[index] = count

        per_topic = (bow_tensor.unsqueeze(0) * torch.log(topics)).sum(1)
        return torch.logsumexp(per_topic, dim=0)

    tiny = Dataset('tiny-0')

    m_prop = InjectedProperty('arbitrary_3x3')
    def score_matrix(d):
        result = []
        for a, b, matrix in tqdm(d[m_prop]):
            result.append([
                [score_pipeline(sen) for sen in row]
                for row in matrix
            ])
        return result

    lda_scores = Property(
        [m_prop],
        'gensim_scores',
        score_matrix
    )

    scores = tiny[lda_scores]
