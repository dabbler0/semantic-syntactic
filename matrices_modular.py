from tools import *
from gpt2 import *
import nltk
from nltk import tokenize
import datasets
from transformers import GPT2Tokenizer
from tqdm import tqdm, trange
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import unidecode
from ngram_model import *
nltk_tokenizer = TreebankWordTokenizer()
nltk_detokenizer = TreebankWordDetokenizer()

IDENTITY = ({ 'type': 'id' }, (lambda x: x))

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
    appearances = {}
    tag_corpus = {}

    for sentence in tqdm(d[sentence_prop]):
        text = nltk_tokenizer.tokenize(unidecode.unidecode(sentence))
        pos = nltk.pos_tag(text)
        for tok, p in pos:
            if tok not in appearances:
                appearances[tok] = set()

            appearances[tok].add(p)

            if p not in tag_corpus:
                tag_corpus[p] = {}

            if tok not in tag_corpus[p]:
                tag_corpus[p][tok] = 0

            tag_corpus[p][tok] += 1

    # Omit ambiguous
    for p in tag_corpus:
        for tok in tag_corpus[p]:
            if len(appearances[tok]) > 1:
                tag_corpus[p][tok] = 0

    return tag_corpus

tagcorpus_prop = Property(
    [sentence_prop],
    'tagcorpus-unambiguous',
    get_tag_corpus,
)

default_replacement_corpus = Dataset('tiny-2')[tagcorpus_prop]
REPLACEABLE_POS_RAW = ('NN', 'NNS', 'NNP', 'JJ', 'RB', 'RBR', 'RBS', 'JJR', 'JJS', 'VBD', 'VB', 'VBG', 'VBN', 'VBZ')

REPLACEABLE_POS = tuple(
    pos for pos in REPLACEABLE_POS_RAW
        if any(default_replacement_corpus[pos][k] > 0 for k in default_replacement_corpus[pos])
)

def choose_from(replacement_dict, exclude):
    keys = list(key for key in replacement_dict.keys() if key != exclude)
    values = torch.Tensor([replacement_dict[key] for key in keys])
    values /= values.sum()

    return np.random.choice(keys, p=values.numpy())

def apply_greenification(s, replacements, markup = False):
    copy = {**s}
    for idx, replacement in replacements:
        if markup:
            if len(copy[idx]) == 4:
                m = {**copy[idx][3]}
            else:
                m = {}
            m['greenified'] = True
            copy[idx] = (
                copy[idx][0],
                replacement,
                copy[idx][2],
                m
            )
        else:
            copy[idx] = (
                copy[idx][0],
                replacement,
                copy[idx][2]
            )
    return copy

def generate_greenification(s,
        replacement_corpus = default_replacement_corpus,
        occupied = set(),
        size_generator = (lambda x: min(1, x))):

    replaceables = set(idx for idx in s if s[idx][2] in REPLACEABLE_POS) - occupied
    num_exchanges = size_generator(len(replaceables))
    subset = np.random.choice(
        list(replaceables),
        num_exchanges,
        replace=False
    )

    return green_for_subset(s, subset, replacement_corpus)

def apply_record(s, record, markup = False):
    if record['type'] == 'green':
        return apply_greenification(s, record['data'], markup)
    elif record['type'] == 'order':
        return apply_permutation(s, record['data'][0], record['data'][1], markup)
    else:
        return s

def apply_permutation(s, subset, permutation, markup = False):
    inverted = { s[i][0]: i for i in s }

    copy = {**s}
    for i, idx in enumerate(subset):
        if markup:
            if len(copy[inverted[idx]]) == 4:
                mk = {**copy[inverted[idx]][3]}
            else:
                mk = {}

            mk['permuted'] = True

            copy[inverted[idx]] = (
                subset[permutation[i]],
                copy[inverted[idx]][1],
                copy[inverted[idx]][2],
                mk
            )
        else:
            copy[inverted[idx]] = (
                subset[permutation[i]],
                copy[inverted[idx]][1],
                copy[inverted[idx]][2]
            )

    return copy

def generate_permutation(s, occupied = set(), size_generator = (lambda x: min(2, x))):
    # Identify
    candidates = set(range(len(s))) - occupied
    subset_size = size_generator(len(candidates))
    subset = np.random.choice(
        list(candidates),
        subset_size,
        replace=False
    )
    # Resample until not the identity
    return permutation_for_subset(subset)

class GreenOrderMutator:
    def __init__(self, op_generator):
        self.op_generator = op_generator

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
        imut_raw, jmut_raw = self.op_generator(s)

        irecord, imut = list(zip(*imut_raw))
        jrecord, jmut = list(zip(*jmut_raw))

        return irecord, jrecord, [
            [
                self.deparse(im(jm(s))) for jm in jmut
            ] for im in imut
        ]

# GENERATING PERMUTATIONS
def gen_nonid_perm(size):
    if size <= 1:
        return None

    permutation = np.random.choice(size, size, replace=False)
    while all(i == j for i, j in enumerate(permutation)):
        permutation = np.random.choice(size, size, replace=False)

    return permutation

def permutation_for_subset(subset):
    permutation = gen_nonid_perm(len(subset))
    if permutation is None:
        return (subset,) + IDENTITY

    lset = list(subset)

    return (
        set(subset),
        { 'type': 'order', 'data': (lset, permutation) },
        (lambda s: apply_permutation(s, lset, permutation))
    )

# GENERATING GREENIFICATIONS
def green_for_subset(s, subset, replacement_corpus = default_replacement_corpus):
    replaceables = set(idx for idx in subset if s[idx][2] in REPLACEABLE_POS)
    replacements = [
        (
            idx,
            choose_from(replacement_corpus[s[idx][2]], s[idx][1])
        )
        for idx in replaceables
    ]

    return (
        set(subset),
        { 'type': 'green', 'data': replacements },
        lambda s: apply_greenification(s, replacements)
    )

# CONTIGUOUS GENERATION
def generate_contiguous_permutations(s, start_index, end_index):
    # Identify
    candidates = set(range(start_index, end_index))
    size_left = len(candidates) // 2
    subset_left = set(np.random.choice(
        list(candidates),
        size_left,
        replace=False
    ))

    subset_right = list(candidates - subset_left)

    return (
        permutation_for_subset(subset_left),
        permutation_for_subset(subset_right)
    )

def generate_contiguous_greenifications(s,
        start_index, end_index,
        replacement_corpus = default_replacement_corpus):

    replaceables = set(idx for idx in range(start_index, end_index) if s[idx][2] in REPLACEABLE_POS)
    num_left = len(replaceables) // 2
    subset_left = set(np.random.choice(
        list(replaceables),
        num_left,
        replace=False
    ))
    subset_right = replaceables - subset_left

    return (
        green_for_subset(s, subset_left, replacement_corpus),
        green_for_subset(s, subset_right, replacement_corpus)
    )

# == 3x3 generator ==

def make_3x3(permutations, greenifications):
    if permutations is None or greenifications is None:
        return [IDENTITY, IDENTITY, IDENTITY], [IDENTITY, IDENTITY, IDENTITY]

    imut = [IDENTITY]
    jmut = [IDENTITY]

    # Two random permutations
    (_, data1, perm1), (_, data2, perm2) = permutations

    p1 = (data1, perm1)
    p2 = (data2, perm2)

    if np.random.random() < 0.5:
        imut.append(p1)
        jmut.append(p2)
    else:
        imut.append(p2)
        jmut.append(p1)

    # Two random greenifications
    (_, data1, green1), (_, data2, green2) = greenifications

    g1 = (data1, green1)
    g2 = (data2, green2)

    if np.random.random() < 0.5:
        imut.append(g1)
        jmut.append(g2)
    else:
        imut.append(g2)
        jmut.append(g1)

    return imut, jmut

def generate_contiguous_3x3(s):
    if len(s) < 6:
        return make_3x3(None, None)

    start, end = sorted(np.random.choice(len(s) - 4, 2))
    end += 4

    return make_3x3(
        generate_contiguous_permutations(s, start_index = start, end_index = end),
        generate_contiguous_greenifications(s, start_index = start, end_index = end)
    )

def generate_arbitrary_3x3(s):
    t_size_generator = (lambda x: np.random.randint(2, x + 1) if x >= 2 else 0)
    g_size_generator = (lambda x: np.random.randint(1, x + 1) if x >= 1 else 0)

    # Two random permutations
    perm1 = generate_permutation(s, size_generator=t_size_generator)
    perm2 = generate_permutation(s, size_generator=t_size_generator)

    # Two random greenifications
    green1 = generate_greenification(s, size_generator=g_size_generator)
    green2 = generate_greenification(s, size_generator=g_size_generator)

    return make_3x3([perm1, perm2], [green1, green2])

def generate_disjoint_3x3(s, t_size_generator, g_size_generator):
    # Two random permutations
    perm1 = generate_permutation(s, size_generator=t_size_generator)
    perm2 = generate_permutation(s, occupied=perm1[0], size_generator=t_size_generator)

    # Two random greenifications
    green1 = generate_greenification(s, size_generator=g_size_generator)
    green2 = generate_greenification(s, occupied=green1[0], size_generator=g_size_generator)

    return make_3x3([perm1, perm2], [green1, green2])

def generate_distant_spans(s, size):
    if len(s) < 2 * size:
        return None

    max_distance = len(s) - 2 * size
    distance = np.random.randint(max_distance + 1)

    init_offset = np.random.randint(len(s) - 2 * size - distance + 1)

    return (
        (init_offset, init_offset + size),
        (init_offset + size + distance, init_offset + 2 * size + distance)
    )

def generate_distant_3x3(s, size = 4):
    spans = generate_distant_spans(s, size)
    if spans is None:
        return make_3x3(None, None)

    return make_3x3(
        [
            permutation_for_subset(set(range(*x))) for x in spans
        ],
        [
            green_for_subset(s, set(range(*x))) for x in spans
        ]
    )

def tokenize_matrix(prop):
    def run_tokenization(d):
        result = []
        for arec, brec, matrix in tqdm(d[prop]):
            result.append([
                [tokenizer(s)['input_ids'] for s in row]
                for row in matrix
            ])
        return result
    return run_tokenization

def tokenized_prop_for(m_prop):
    return Property(
        [m_prop],
        m_prop.name + '_tok',
        tokenize_matrix(m_prop)
    )

criterion = torch.nn.CrossEntropyLoss(reduction='sum')
def score(t):
    with torch.no_grad():
        t = torch.LongTensor(t).unsqueeze(0).cuda()
        result = lmmodel(t).logits[0]
        return criterion(result[:-1], t[0][1:]).cpu().numpy().tolist()

def lstm_scorer(model):
    model.eval()
    def scorer(t):
        with torch.no_grad():
            t = torch.LongTensor(t)
            inp = torch.nn.utils.rnn.pack_sequence([t[:-1]]).cuda()
            targ = torch.nn.utils.rnn.pack_sequence([t[1:]]).cuda()

            result = model(inp)
            return criterion(result, targ).cpu().numpy().tolist()
    return scorer

def score_matrix(prop, score_fn = score):
    def run_scoring(d):
        result = []
        for matrix in tqdm(d[prop]):
            result.append([
                [score_fn(tokens) if 0 < len(tokens) < 1024 else None for tokens in row]
                for row in matrix
            ])
        return result
    return run_scoring

def scored_prop_for(m_prop, score_fn = score, score_label = 'scored'):
    t_prop = tokenized_prop_for(m_prop)

    return Property(
        [t_prop],
        m_prop.name + '_%s' % score_label,
        score_matrix(t_prop),
        version = 1
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

def mut_score_for(m):
    normalizer = torch.logsumexp(m.reshape(-1), 0)
    m = torch.exp(m - normalizer)
    Da, Db = m.sum(0), m.sum(1)
    Ha, Hb = torch.sum(Da * torch.log(Da)), torch.sum(Db * torch.log(Db))
    Hab = torch.sum(m * torch.log(m))

    return Ha + Hb - Hab

def abs_score_for(m):
    return abs(
        m[0, 1] + m[1, 0] - m[0, 0] - m[1, 1]
    )

def nuc_score_for(m):
    t = torch.exp(torch.Tensor(m))
    return torch.norm(t / torch.norm(t, p='fro'), p='nuc')

def determine_subset_for(mod):
    if mod['type'] == 'id':
        return set()
    elif mod['type'] == 'order':
        return set(mod['data'][0])
    elif mod['type'] == 'green':
        return set(x[0] for x in mod['data'])

def determine_distance_for_pair(mod1, mod2):
    if len(determine_subset_for(mod1)) == 0 or len(determine_subset_for(mod2)) == 0:
        return 0

    end1 = max(*determine_subset_for(mod1))
    beg2 = min(*determine_subset_for(mod2))
    if end1 > beg2:
        end2 = max(*determine_subset_for(mod2))
        beg1 = min(*determine_subset_for(mod1))

        if end2 > beg1:
            return 0
        else:
            return beg1 - end2
    else:
        return beg2 - end1

def four_minor_scores(d, p, f = abs_score_for):
    result = []
    for matrix in tqdm(d[p]):
        if any(score is None for row in matrix for score in row):
            result.append(None)
        else:
            result.append(tuple(
                f(minor) for minor in four_minors(matrix)
            ))
    return result

def entanglement_model(s_prop, i, j, model_factory, model_name,
        epochs = 13):

    BATCH_SIZE = 128
    model = model_factory().cuda()

    def train(d):
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        dataset = d[s_prop]

        batched_dataset = [
            dataset[k:k+BATCH_SIZE]
            for k in range(len(dataset) // BATCH_SIZE + 1)
        ]

        # r, a, b, d
        batches = [
            (
                torch.Tensor([x[0][0] for x in b]),
                torch.Tensor([x[i][0] for x in b]),
                torch.Tensor([x[0][j] for x in b]),
                torch.Tensor([x[i][j] for x in b])
            ) for b in batched_dataset
        ]

        criterion = torch.nn.MSELoss()

        for _ in trange(epochs):
            # Rebatch
            np.random.shuffle(batches)

            # Single-S model
            for r, a, b, d in batches:
                # Inputs a:
                prediction = model(r.cuda(), a.cuda(), b.cuda())
                loss = criterion(prediction, d.cuda())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

    return Property(
        [s_prop],
        '%s_%d_%d_model_%s_%s' % (s_prop.name, i, j, model_name, epochs),
        train
    )

def four_score_prop(m_prop, model_fn = score, model_label = 'scored', score_type = 'abs'):
    scoring_function = ({
        'abs': abs_score_for,
        'nuc': nuc_score_for,
        'mut': mut_score_for
    })[score_type]
    s_prop = scored_prop_for(m_prop, model_fn, model_label)

    return Property(
        [s_prop],
        '%s_%s_%s_evals' % (m_prop.name, model_label, score_type),
        (lambda d: four_minor_scores(d, s_prop, f = scoring_function)),
        version = 1
    )
