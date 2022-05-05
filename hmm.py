import torch
from tools import *
from languages import *
from tqdm import tqdm, trange

eps = 1e-4

class HMM:
    def __init__(self, unigrams, ngrams, emissions, device = 'cuda:0', smoothing = 0.1):
        self.unigrams = unigrams
        self.ngrams = ngrams
        self.device = device

        self.n = len(next(iter(ngrams.keys())))

        # n-1 prefixes
        self.mgrams = {}
        for ngram in self.ngrams:
            mgram = ngram[:-1]
            if mgram not in self.mgrams:
                self.mgrams[mgram] = 0
            self.mgrams[mgram] += self.ngrams[ngram]

        self.id_to_token = list(self.unigrams.keys())
        self.token_to_id = { t: i for i, t in enumerate(self.id_to_token) }

        self.id_to_state = list(self.mgrams.keys())
        self.state_to_id = { s: i for i, s in enumerate(self.id_to_state) }

        self.init_state_counts = torch.Tensor([self.mgrams[s] for s in self.id_to_state]).to(device)
        self.default_vector = torch.log(self.init_state_counts / self.init_state_counts.sum())

        self.totals = { m: sum(self.ngrams[ngram] for ngram in self.ngrams if ngram[:-1] == m) for m in self.mgrams }
        self.transitions = {
            s: {
                u: (self.ngrams[s + (u,)] / self.mgrams[s] if s + (u,) in self.ngrams else 0)
                for u in self.unigrams
            }
            for s in self.mgrams
        }

        self.transition_matrix = torch.Tensor([
            [
                self.transitions[m1][m2[-1]] if m2[:-1] == m1[1:] and m2[-1] in self.transitions[m1] else eps
                for m2 in self.id_to_state
            ] for m1 in self.id_to_state
        ]).to(device)

        self.transition_matrix = torch.log(self.transition_matrix)

        self.emissions = emissions

        self.emission_vectors = {}

        self.mgram_totals = torch.zeros(len(self.mgrams)).to(device)

        for category in self.emissions:
            for token in self.emissions[category]:
                if token not in self.emission_vectors:
                    self.emission_vectors[token] = torch.zeros(len(self.mgrams)).to(device)

                for mgram in self.mgrams:
                    if mgram[-1] == category:
                        self.mgram_totals[self.state_to_id[mgram]] += self.emissions[category][token]
                        self.emission_vectors[token][self.state_to_id[mgram]] = \
                                self.emissions[category][token]

        vocab_size = len(self.emission_vectors)

        # By-hand smoothing for OOV event
        oov_prob = torch.Tensor([1 / (1024 * 1024)]).to(device)
        log_noov_prob = torch.log(1 - oov_prob)
        log_oov_prob = torch.log(oov_prob)

        self.normal_emission_vectors = {
            t: torch.log(
                (v + smoothing) /
                (self.mgram_totals + smoothing * vocab_size)
            ) + log_noov_prob
            for t, v in self.emission_vectors.items()
        }

        self.unknown_vector = log_oov_prob.expand((len(self.mgrams),)).to(device)

    def conditional_entropy(self, tokens):
        # Running state of
        # P(sen | last token is H)

        p = torch.exp(self.default_vector)
        z = self.default_vector * torch.exp(self.default_vector)

        for token in tokens:
            z = (z.unsqueeze(1) + p * self.pxtt(token)) * torch.exp(self.transition_matrix)
            p = (p.unsqueeze(1) * torch.exp(self.transition_matrix)).sum(0)

        return z.sum()

    def pxtt(self, x):#, t, state):
        # transition_matrix[old][new] = P(old | new)
        # => pre_normalized[old][new] = P(old | new) * P(tok | new)
        pre_normalized = self.transition_matrix + self.normal_emission_vectors[x].unsqueeze(0)
        normalized = torch.softmax(pre_normalized, dim=1)
        return normalized

    def tag(self, tokens):
        state = self.default_vector
        #sequences = [(state[-1],) for state in self.id_to_state]

        for i, token in enumerate(tokens):
            emit_state = state + (self.normal_emission_vectors[token]
                    if token in self.normal_emission_vectors else self.unknown_vector)

            if i == len(tokens) - 1:
                result, index = torch.max(emit_state, dim=0)
                return result#, sequences[index]

            trellis = emit_state.unsqueeze(1) + self.transition_matrix
            # Trellis now has: row = prev state, col = new state
            # trellis[row][col] = joint prob
            # want maximal tag sequence, so update best_gen_prob[col] = max(trellis[:, col])
            new_state, indices = torch.max(trellis, dim=0)
            state = new_state
            #sequences = [sequences[idx] + (self.id_to_state[i][-1],) for i, idx in enumerate(indices)]

    def score(self, tokens):
        # Running state of
        # P(sen | last token is H)

        state = self.default_vector

        for token in tokens:
            emit_state = state + (self.normal_emission_vectors[token]
                    if token in self.normal_emission_vectors else self.unknown_vector)
            trellis = emit_state.unsqueeze(1) + self.transition_matrix
            new_state = torch.logsumexp(trellis, dim=0)
            state = new_state

        return torch.logsumexp(state, dim=0)

sentence_prop = InjectedProperty('sentence')

ENGLISH = StanfordMosesToolset('en')
ENGLISH.__enter__()

def get_hmm_params(corpus, language=ENGLISH):
    unigrams = {}
    bigrams = {}
    emissions = {}

    for sentence in tqdm(corpus):#d[sentence_prop]):
        text = language.tokenize(sentence)
        pos = language.pos_tag(text)
        for tok, p in pos:
            # Emissions
            if p not in emissions:
                emissions[p] = {}

            if tok not in emissions[p]:
                emissions[p][tok] = 0

            emissions[p][tok] += 1

        pos_sequence = [p for tok, p in pos]

        # Unigram count
        for p in pos_sequence:
            if p not in unigrams:
                unigrams[p] = 0
            unigrams[p] += 1

        # Bigram count
        for bg in nltk.bigrams(pos_sequence):
            if bg not in bigrams:
                bigrams[bg] = 0
            bigrams[bg] += 1

    return unigrams, bigrams, emissions

'''
hmm_prop = Property(
    [sentence_prop],
    'pos-hmm-params',
    get_hmm_params
)
'''

if __name__ == '__main__':
    tiny2 = Dataset('tiny-1')
    tiny = Dataset('tiny-0')
    union_dataset = tiny[sentence_prop] + tiny2[sentence_prop]
    '''
    unigrams, bigrams, emissions = get_hmm_params(union_dataset)

    torch.save((unigrams, bigrams, emissions), 'hmm-stanza.pt')
    '''
    unigrams, bigrams, emissions = torch.load('hmm-stanza.pt')

    model = HMM(unigrams, bigrams, emissions)


    m_prop = InjectedProperty('arbitrary_3x3_stanza')

    def tokenize_matrix(d):
        result = []
        for a, b, matrix in tqdm(d[m_prop]):
            result.append([
                [ENGLISH.tokenize(sen) for sen in row]
                for row in matrix
            ])
        return result

    wtok_prop = Property(
        [m_prop],
        'arbitrary_3x3_stanza_wtok',
        tokenize_matrix
    )

    def score_matrix(d):
        result = []
        for matrix in tqdm(d[wtok_prop]):
            result.append([
                [model.tag(sen) for sen in row]
                for row in matrix
            ])
        return result

    hmm_scores = Property(
        [wtok_prop],
        'hmm_scores_tag_prob3',
        score_matrix
    )

    scores = tiny[hmm_scores]
    '''

    while True:
        s = input('>> ')
        print('Tag: %f, %r' % model.tag(ENGLISH_NLTK.tokenize(s)))
        print('Conditional entropy: %f' % model.conditional_entropy(ENGLISH_NLTK.tokenize(s)))

    '''
