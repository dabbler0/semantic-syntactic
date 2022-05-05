from tools import *
from languages import *
from tqdm import tqdm, trange
from itertools import chain
import string

ONE = torch.Tensor([1]).cuda()

translation = str.maketrans('', '', string.punctuation)
def normalize_str(s):
    return s.lower().translate(translation).strip()

class TopicModel:
    def __init__(self, topics = 64, vocab = 65535, smoothing_constant = 0):
        self.topic_matrix = torch.log(
            torch.softmax(torch.randn(topics, vocab), dim=1)
        ).cuda()
        self.prior = torch.softmax(torch.randn(topics), dim=0).cuda()
        self.smoothing_constant = 0.1

    def export(self):
        return self.topic_matrix, self.prior

    def load(self, topic_matrix, prior):
        self.topic_matrix = topic_matrix
        self.prior = prior

    def score(self, bow):
        log_probs = self.topic_matrix @ bow + torch.log(self.prior)
        return torch.logsumexp(log_probs, dim=0)

    def batched_em_update(self, corpus, report_score = True, total_len = 1):
        running_totals = torch.zeros_like(self.topic_matrix).cuda()
        running_prior = torch.zeros_like(self.prior).cuda()

        running_score = 0

        log_prior = torch.log(self.prior)

        # Estimate
        for batch in tqdm(corpus):
            batch = batch.cuda()

            # batch will be batch_size x vocab
            # topic matrix is topics x vocab
            log_probs = batch @ self.topic_matrix.t()

            # Now log probs is batch_size x topics

            if report_score:
                running_score += (torch.logsumexp(log_prior.unsqueeze(0) + log_probs, dim=1) / torch.maximum(ONE, batch.sum(1))).sum() / total_len

            topic_preds = torch.softmax(log_probs, dim=1)
            running_totals += topic_preds.t() @ batch
            running_prior += topic_preds.sum(0)

        # Smoothing
        running_totals += self.smoothing_constant

        # Maximize
        self.topic_matrix = torch.log(running_totals / running_totals.sum(1, keepdim=True))
        self.prior = running_prior

        if report_score:
            return running_score

    def em_update(self, corpus, processor, report_score = True):
        running_totals = torch.zeros_like(self.topic_matrix).cuda()
        running_prior = torch.zeros_like(self.prior).cuda()

        running_score = 0

        log_prior = torch.log(self.prior)

        # Estimate
        for document in tqdm(corpus):
            bow_vector = processor(document)
            log_probs = self.topic_matrix @ bow_vector

            if report_score:
                running_score += torch.logsumexp(log_prior + log_probs, dim=0) / (len(document) * len(corpus))

            topic_preds = torch.softmax(log_probs, dim=0)
            running_totals += topic_preds.unsqueeze(1) * bow_vector.unsqueeze(0)
            running_prior += topic_preds

        # Smoothing
        running_totals += self.smoothing_constant

        # Maximize
        self.topic_matrix = torch.log(running_totals / running.totals.sum(1))
        self.prior = running_prior

        if report_score:
            return running_score

ENGLISH = StanfordMosesToolset('en')
ENGLISH.__enter__()

sentence_prop = InjectedProperty('sentence')
def get_wordtokens(d, language=ENGLISH):
    return [
        tuple(language.tokenize(sentence))
        for sentence in tqdm(d[sentence_prop])
    ]

wordtoken_prop = Property(
    [sentence_prop],
    'wordtokens3',
    get_wordtokens
)

if __name__ == '__main__':
    tiny1 = Dataset('tiny-1')
    tiny0 = Dataset('tiny-0')

    union_corpus = tiny1[wordtoken_prop] + tiny0[wordtoken_prop]

    print('Retreived corpus.')

    vocabulary = set(normalize_str(x) for x in chain(*union_corpus))
    print('Got vocabulary (size %d).' % len(vocabulary))
    id2word = list(vocabulary)
    word2id = { t: i for i, t in enumerate(id2word) }
    print('Made dictionary.')

    def process(words):
        result = torch.zeros(len(vocabulary))
        for word in words:
            word = normalize_str(word)
            if word != '':
                result[word2id[word]] += 1
        return result

    BSIZE = 128
    batched_corpus = [
        torch.stack([
            process(doc)
            for doc in union_corpus[i * BSIZE : (i + 1) * BSIZE]
        ]) for i in trange(len(union_corpus) // BSIZE + 1)
    ]

    print('Processed entire corpus into %d batches.' % len(batched_corpus))

    model = TopicModel(vocab = len(vocabulary))

    for epoch in range(32):
        score = model.batched_em_update(batched_corpus, report_score = True, total_len = len(union_corpus))
        print('Finished epoch %d. Mean per-token perplexity: %f' % (epoch, score))

    torch.save(id2word, 'topic-vocab-stanza-nopunct.pt')
    torch.save(model.export(), 'topic-model-stanza-nopunct.pt')
