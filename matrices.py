from tools import *
from gpt2 import *
import nltk
from nltk import tokenize
import datasets
from transformers import GPT2Tokenizer
from tqdm import tqdm
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import unidecode
from ngram_model import *
nltk_tokenizer = TreebankWordTokenizer()
nltk_detokenizer = TreebankWordDetokenizer()

class MatrixProp(Property):
    def __init__(self, sentence_prop, name, mutator, version = 0):
        self.mutator = mutator
        super(MatrixProp, self).__init__(
            [sentence_prop],
            name,
            (lambda d: self.gen_mutations(d[sentence_prop])),
            version
        )

    def gen_mutations(self, l):
        result = []
        for s in tqdm(l):
            result.append(self.mutator(s))
        return result

sentence_prop = InjectedProperty('sentence')
def get_tag_corpus(d):
    tag_corpus = {}
    for sentence in tqdm(d[sentence_prop]):
        text = nltk_tokenizer.tokenize(unidecode.unidecode(sentence))
        pos = nltk.pos_tag(text)
        for tok, p in pos:
            if p not in tag_corpus:
                tag_corpus[p] = []
            tag_corpus[p].append(tok)
    return tag_corpus

tagcorpus_prop = Property(
    [sentence_prop],
    'tagcorpus',
    get_tag_corpus,
    version = 1
)

default_replacement_corpus = Dataset('tiny-2')[tagcorpus_prop]
REPLACEABLE_POS = ('NN', 'NNS', 'NNP', 'JJ', 'VBD', 'VBZ')

class IdentityMutator:
    def __init__(self):
        return

    def generate(self, s):
        return { 'type': 'id' }, (lambda x: x)

class GreenMutator:
    def __init__(self, num_exchanges, replacement_corpus = default_replacement_corpus):
        self.num_exchanges = num_exchanges
        self.replacement_corpus = replacement_corpus

    def generate(self, s):
        replaceables = [idx for idx in s if s[idx][2] in REPLACEABLE_POS]
        num_exchanges = min(len(replaceables), self.num_exchanges())
        subset = np.random.choice(
            len(replaceables),
            num_exchanges, replace=False)

        replacements = [
            (
                replaceables[idx],
                np.random.choice(self.replacement_corpus[s[replaceables[idx]][2]])
            )
            for idx in subset
        ]

        def mutate(s):
            copy = {**s}
            for idx, replacement in replacements:
                copy[idx] = (
                    copy[idx][0],
                    replacement,
                    copy[idx][2]
                )
            return copy

        return { 'type': 'green', 'data': replacements }, mutate

class OrderMutator:
    def __init__(self, subset_size):
        self.subset_size = subset_size

    def generate(self, s):
        # Identify
        subset_size = min(len(s), self.subset_size())
        subset = np.random.choice(len(s), subset_size, replace=False)
        permutation = np.random.choice(subset_size, subset_size, replace=False)

        def mutate(s):
            inverted = { s[i][0]: i for i in s }

            copy = {**s}
            for i, idx in enumerate(subset):
                copy[inverted[idx]] = (
                    subset[permutation[i]],
                    copy[inverted[idx]][1],
                    copy[inverted[idx]][2]
                )

            return copy

        return { 'type': 'order', 'data': (subset, permutation) }, mutate

class GreenOrderMutator:
    def __init__(self, imut, jmut):
        self.imut = imut
        self.jmut = jmut

    def parse(self, s):
        return {
            i: (i, tok, pos) for i, (tok, pos) in enumerate(
                nltk.pos_tag(nltk_tokenizer.tokenize(unidecode.unidecode(s)))
            )
        }

    def deparse(self, s):
        inverted = {
            s[i][0]: s[i][1]
            for i in s
        }
        flattened = [inverted[i] for i in range(len(inverted))]
        return nltk_detokenizer.detokenize(flattened)

    def __call__(self, s):
        s = self.parse(s)
        irecord, imut = zip(*[mut.generate(s) for mut in self.imut])
        jrecord, jmut = zip(*[mut.generate(s) for mut in self.jmut])

        return irecord, jrecord, [
            [
                self.deparse(im(jm(s))) for jm in jmut
            ] for im in imut
        ]

mutator_3x3 = GreenOrderMutator(
    [IdentityMutator(), OrderMutator(lambda: np.random.randint(2, 16)), GreenMutator(lambda: np.random.randint(1, 16))],
    [IdentityMutator(), OrderMutator(lambda: np.random.randint(2, 16)), GreenMutator(lambda: np.random.randint(1, 16))]
)

matrix_3x3_prop = MatrixProp(
    sentence_prop,
    '3x3_one_each',
    mutator_3x3,
    version = 3
)

def tokenize_matrix(d):
    result = []
    for arec, brec, matrix in tqdm(d[matrix_3x3_prop]):
        result.append([
            [tokenizer(s)['input_ids'] for s in row]
            for row in matrix
        ])
    return result

def tokenize_single(d):
    result = []
    for s in tqdm(d[sentence_prop]):
        result.append(tokenizer(s)['input_ids'])
    return result

def tokenized_prop_for(s_prop):
    return Property(
        [s_prop],
        s_prop.name + '_tokenized',
        tokenize_single
    )

tokenized_prop = Property(
    [sentence_prop],
    'tokenized',
    tokenize_single,
    version = 1
)

def tokenized_3x3_prop_for(m_prop):
    return Property(
        [m_prop],
        m_prop.name + '_tok',
        tokenize_matrix
    )

tokenized_3x3_prop = Property(
    [matrix_3x3_prop],
    '3x3_one_each_tok',
    tokenize_matrix,
    version = 3
)

# Ngram counts

def get_ngram_counts(d, horizon=3):
    return NgramModel(d[tokenized_prop])

criterion = torch.nn.CrossEntropyLoss()
def score(t):
    with torch.no_grad():
        t = torch.LongTensor(t).unsqueeze(0).cuda()
        result = lmmodel(t).logits[0]
        return criterion(result[:-1], t[0][1:], reduction='sum').cpu().numpy().tolist()

def score_matrix(d):
    result = []
    for matrix in tqdm(d[tokenized_3x3_prop]):
        result.append([
            [score(tokens) if 0 < len(tokens) < 1024 else None for tokens in row]
            for row in matrix
        ])
    return result

scored_3x3_prop = Property(
    [tokenized_3x3_prop],
    '3x3_one_each_score',
    score_matrix,
    version = 2
)

ngram_count_prop = Property(
    [tokenized_prop],
    'ngram_counts',
    get_ngram_counts,
    version = 4
)

def ngram_matrix(d1, d2):
    model = d2[ngram_count_prop]

    result = []
    for matrix in tqdm(d1[tokenized_3x3_prop]):
        result.append([
            [model.score(tokens) if 0 < len(tokens) < 1024 else None for tokens in row]
            for row in matrix
        ])
    return result

def ngram_3x3_prop(d2):
    return Property(
        [tokenized_3x3_prop, ngram_count_prop],
        '3x3_one_each_ngram_%s' % d2.name,
        (lambda d: ngram_matrix(d, d2)),
        version = 4
    )

def four_minors(t):
    t = torch.Tensor(t)
    return [
        # Order-order
        torch.Tensor([
            [t[0, 0], t[0, 1]],
            [t[1, 0], t[1, 1]]
        ]),
        # Green-Green
        torch.Tensor([
            [t[0, 0], t[0, 2]],
            [t[2, 0], t[2, 2]]
        ]),
        # Mixed 1
        torch.Tensor([
            [t[0, 0], t[0, 2]],
            [t[1, 0], t[1, 2]]
        ]),
        # Mixed 2
        torch.Tensor([
            [t[0, 0], t[0, 1]],
            [t[2, 0], t[2, 1]]
        ])
    ]

def abs_score_for(m):
    return abs(
        m[0, 1] + m[1, 0] - m[0, 0] - m[1, 1]
    )

def nuclear_norm_score_for(m):
    t = torch.exp(torch.Tensor(m))
    return torch.norm(t / torch.norm(t, p='fro'), p='nuc')

def four_minor_scores(d, p = scored_3x3_prop, f = abs_score_for):
    result = []
    for matrix in tqdm(d[p]):
        if any(score is None for row in matrix for score in row):
            result.append(None)
        else:
            result.append(tuple(
                f(minor) for minor in four_minors(matrix)
            ))
    return result

gpt_four_score_prop = Property(
    [scored_3x3_prop],
    '3x3_one_each_minors',
    (lambda d: four_minor_scores(d, f = abs_score_for)),
    version = 3
)

gpt_nuc_score_prop = Property(
    [scored_3x3_prop],
    '3x3_one_each_nuc',
    (lambda d: four_minor_scores(d, f = abs_score_for)),
    version = 2
)

def four_score_prop(p):
    return Property(
        [p],
        '%s_minors' % p.name,
        (lambda d: four_minor_scores(d, p = p, f = abs_score_for)),
        version = 6
    )
