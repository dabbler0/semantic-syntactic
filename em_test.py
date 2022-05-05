from tools import *
from languages import *
from tqdm import tqdm, trange
from itertools import chain
from em_nb_model import *


if __name__ == '__main__':
    id2word = torch.load('topic-vocab-stanza.pt')
    word2id = { t: i for i, t in enumerate(id2word) }

    params = torch.load('topic-model-stanza.pt')
    model = TopicModel()
    model.load(*params)

    def process(words):
        result = torch.zeros(len(id2word))
        for word in words:
            word = normalize_str(word)
            if word.strip() == '':
                continue

            if word in word2id:
                wid = word2id[word]
                result[wid] += 1
        return result

    def score(sentence):
        bow = process(ENGLISH_NLTK.tokenize(sentence))
        return model.score(bow.cuda()).cpu().numpy().tolist()

    m_prop = InjectedProperty('arbitrary_3x3_stanza')

    def score_matrix(d):
        result = []
        for a, b, matrix in tqdm(d[m_prop]):
            result.append([
                [score(sen) for sen in row]
                for row in matrix
            ])
        return result

    em_scores = Property(
        [m_prop],
        'topic_scores_stanza_nopunct',
        score_matrix
    )

    tiny = Dataset('tiny-0')
    scores = tiny[em_scores]
