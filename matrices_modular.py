from tools import *
from languages import *
from gpt2 import *
import nltk
from nltk import tokenize
import datasets
from transformers import GPT2Tokenizer
from tqdm import tqdm, trange
import numpy as np
from entanglement_models import *
from ngram_model import *

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
def get_tag_corpus(d, language=ENGLISH_NLTK):
    appearances = {}
    tag_corpus = {}

    for sentence in tqdm(d[sentence_prop]):
        text = language.tokenize(sentence)
        pos = language.pos_tag(text)
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

def get_untagged_corpus(d, language=ENGLISH_NLTK):
    results = {}

    for sentence in tqdm(d[sentence_prop]):
        text = language.tokenize(sentence)
        for tok in text:
            if tok not in results:
                results[tok] = 0
            results[tok] += 1

    return results

untag_prop = Property(
    [sentence_prop],
    'untagged-words',
    get_untagged_corpus
)

tagcorpus_prop = Property(
    [sentence_prop],
    'tagcorpus-unambiguous',
    get_tag_corpus,
)

def make_unigram_pool(untag):
    keys = list(key for key in untag.keys())
    values = torch.Tensor([untag[key] for key in keys])
    values /= values.sum()

    return keys, values

default_replacement_corpus = Dataset('tiny-2')[tagcorpus_prop]
default_untagged_corpus = make_unigram_pool(Dataset('tiny-2')[untag_prop])
REPLACEABLE_POS_RAW = ('NN', 'NNS', 'NNP', 'JJ', 'RB', 'RBR', 'RBS', 'JJR', 'JJS', 'VBD', 'VB', 'VBG', 'VBN', 'VBZ')

REPLACEABLE_POS = tuple(
    pos for pos in REPLACEABLE_POS_RAW
        if any(default_replacement_corpus[pos][k] > 0 for k in default_replacement_corpus[pos])
)

class Replacer:
    def __init__(self, replacement_corpus):
        self.replacement_corpus = replacement_corpus
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.cache = {}

    def pos_to_wordnet(self, pos):
        return (pos[0].lower() if pos[0] in ('N', 'V') else None)

    def candidates(self, word, pos):
        wpos = self.pos_to_wordnet(pos)

        if wpos is None:
            return []

        if (word, wpos) in self.cache:
            return self.cache[word, wpos]

        lemma = self.lemmatizer.lemmatize(word, pos=wpos)
        candidates = []

        for key in self.replacement_corpus:
            if self.pos_to_wordnet(key) == wpos:
                for other in self.replacement_corpus[key]:
                    if (self.lemmatizer.lemmatize(other, pos=wpos) == lemma and
                            other != word):
                        candidates.append(other)

        self.cache[word, wpos] = candidates

        return candidates

default_replacer = Replacer(default_replacement_corpus)

def draw_unigram(keys, values, n):
    if n == 0:
        return []
    else:
        return [keys[t] for t in values.multinomial(n)]

def choose_from(replacement_dict, exclude):
    keys = list(key for key in replacement_dict.keys() if key != exclude)
    values = torch.Tensor([replacement_dict[key] for key in keys])
    values /= values.sum()

    return np.random.choice(keys, p=values.numpy())

def apply_morphology(s, idx, replacement, markup = False):
    copy = {**s}

    if markup:
        if len(copy[idx]) == 4:
            m = {**copy[idx][3]}
        else:
            m = {}
        m['morphed'] = True
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

def deep_replace(tpl, string):
    if type(tpl) == str:
        return string
    else:
        return tuple(deep_replace(sub, string) for sub in tpl)

def apply_drops(s, indices, markup = False):
    copy = {**s}
    for idx in indices:
        if markup:
            if len(copy[idx]) == 4:
                m = {**copy[idx][3]}
            else:
                m = {}

            m['dropped'] = True
            copy[idx] = (
                copy[idx][0],
                tuple(),
                copy[idx][2],
                m
            )
        else:
            copy[idx] = (
                copy[idx][0],
                tuple(),
                copy[idx][2]
            )
    return copy

def apply_duplications(s, indices, markup = False):
    copy = {**s}
    for idx in indices:
        if markup:
            if len(copy[idx]) == 4:
                m = {**copy[idx][3]}
            else:
                m = {}

            m['duplicated'] = True
            copy[idx] = (
                copy[idx][0],
                (copy[idx][1], copy[idx][1]),
                copy[idx][2],
                m
            )
        else:
            copy[idx] = (
                copy[idx][0],
                (copy[idx][1], copy[idx][1]),
                copy[idx][2]
            )
    return copy

def apply_greenification(s, replacements, markup = False):
    copy = {**s}
    for idx, replacement in replacements:
        if markup:
            if len(copy[idx]) == 4:
                m = {**copy[idx][3]}
            else:
                m = {}

            if type(copy[idx][1]) == tuple:
                replacement = deep_replace(copy[idx][1], replacement)

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
        replacement_filter = (lambda x: x in REPLACEABLE_POS),
        occupied = set(),
        size_generator = (lambda x: min(1, x))):

    replaceables = set(idx for idx in s if replacement_filter(s[idx][2])) - occupied
    num_exchanges = size_generator(len(replaceables))
    subset = np.random.choice(
        list(replaceables),
        num_exchanges,
        replace=False
    )

    return green_for_subset(s, subset, replacement_corpus, replacement_filter)

def generate_drops(s,
        pos_filter = (lambda x: True),
        size_generator = (lambda x: min(1, x))):

    candidates = [idx for idx in s if pos_filter(s[idx][2])]

    num_exchanges = size_generator(len(candidates))
    subset = np.random.choice(
        list(candidates),
        num_exchanges,
        replace=False
    )

    return drops_for_subset(s, subset)

def generate_duplications(s,
        pos_filter = (lambda x: True),
        size_generator = (lambda x: min(1, x))):

    candidates = [idx for idx in s if pos_filter(s[idx][2])]

    num_exchanges = size_generator(len(candidates))
    subset = np.random.choice(
        list(candidates),
        num_exchanges,
        replace=False
    )

    return duplications_for_subset(s, subset)

def generate_single_morph(s,
        replacer = default_replacer):

    trial_order = list(range(len(s)))
    np.random.shuffle(trial_order)

    for idx in trial_order:
        result = morph_for_index(s, idx, replacer = replacer)
        if result is not None:
            return result

    return None

def generate_random(s,
        untag_corpus = default_untagged_corpus,
        size_generator = (lambda x: min(1, x))):

    num_exchanges = size_generator(len(s))
    subset = np.random.choice(
        list(range(len(s))),
        num_exchanges,
        replace=False
    )

    return replace_for_subset(s, subset, untag_corpus)

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

def fully_flatten(arr):
    for tok in arr:
        if type(tok) == tuple or type(tok) == list:
            yield from fully_flatten(tok)
        else:
            yield tok

class Mutator:
    def __init__(self, language = ENGLISH_NLTK):
        self.language = language

    def parse(self, s):
        return {
            i: (i, tok, pos) for i, (tok, pos) in enumerate(
                self.language.pos_tag(
                    self.language.tokenize(s)
                )
            )
        }

    def deparse(self, s):
        inverted = {
            s[i][0]: s[i][1]
            for i in s
        }
        flattened = list(fully_flatten([inverted[i] for i in range(len(inverted))]))
        # Unfold any duplicated tokens
        return self.language.detokenize(flattened)

    # Testing purposes
    def test_generator(self, s, generator):
        s = self.parse(s)
        new_s = generator(s)[2](s)
        return self.deparse(new_s)

    def generate_single_matrix(self, raw_s, generators, full_score):
        s = self.parse(raw_s)
        mutators = [('identity', (set(),) + IDENTITY)]
        for generator, count, tag in generators:
            for _ in range(count):
                mutators.append(
                    (tag, generator(s))
                )

        # Create matrix
        score_matrix = []
        string_matrix = []
        for tag_a, a in tqdm(mutators):
            a = a[2]
            score_row = []
            string_row = []
            for tag_b, b in mutators:
                b = b[2]
                string = self.deparse(a(b(s)))
                string_row.append(string)

                score_row.append(full_score(string))
            score_matrix.append(score_row)
            string_matrix.append(string_row)

        return mutators, string_matrix, score_matrix

def all_subsets(l):
    if len(l) == 0:
        yield []
        return

    for sub in all_subsets(l[1:]):
        yield [l[0]] + sub
        yield sub

class AllSubsetsMutator(Mutator):
    def __init__(self, op_generator, language=ENGLISH_NLTK):
        super(AllSubsetsMutator, self).__init__(language)
        self.op_generator = op_generator

    def __call__(self, s):
        s = self.parse(s)

        operations = self.op_generator(s)

        records = []
        result = []

        # Iterate over subsets
        for subset in all_subsets(operations):
            cursor = s
            ops_applied = []
            for span, record, op in subset:
                cursor = op(cursor)
                ops_applied.append(record)
            records.append(ops_applied)
            result.append(self.deparse(cursor))

        return [IDENTITY[0]], records, [result]

class GreenOrderMutator(Mutator):
    def __init__(self, op_generator, language=ENGLISH_NLTK):
        super(GreenOrderMutator, self).__init__(language)
        self.op_generator = op_generator

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

def duplications_for_subset(s, subset):
    return (
        set(subset),
        { 'type': 'duplication', 'data': list(subset) },
        (lambda s: apply_duplications(s, subset))
    )

def drops_for_subset(s, subset):
    return (
        set(subset),
        { 'type': 'drops', 'data': list(subset) },
        (lambda s: apply_drops(s, subset))
    )

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

def morph_for_index(s, idx, replacer = default_replacer):
    candidates = replacer.candidates(s[idx][1], s[idx][2])

    if len(candidates) == 0:
        return None

    replacement = np.random.choice(candidates)

    return (
        {idx},
        { 'type': 'morph', 'data': (idx, replacement) },
        (lambda s: apply_morphology(s, idx, replacement))
    )


# GENERATING GREENIFICATIONS
def green_for_subset(s, subset, replacement_corpus = default_replacement_corpus, replacement_filter = (lambda x: x in REPLACEABLE_POS)):
    replaceables = set(idx for idx in subset if replacement_filter(s[idx][2]))
    replacements = []
    for idx in replaceables:
        replacement = choose_from(replacement_corpus[s[idx][2]], s[idx][1])
        replacements.append(
            (idx, replacement)
        )

    return (
        set(subset),
        { 'type': 'green', 'data': replacements },
        lambda s: apply_greenification(s, replacements)
    )

# GENERATING RANDOM REPLACEMENTS
def replace_for_subset(s, subset, untag_corpus = default_untagged_corpus):
    replacements = list(zip(
        subset, # Note order doesn't matter here
        draw_unigram(*untag_corpus, len(subset))
    ))

    return (
        set(subset),
        { 'type': 'green', 'data': replacements, 'is-random': True },
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
        replacement_filter = (lambda x: x in REPLACEABLE_POS),
        replacement_corpus = default_replacement_corpus):

    replaceables = set(idx for idx in range(start_index, end_index) if replacement_filter(s[idx][2]))
    num_left = len(replaceables) // 2
    subset_left = set(np.random.choice(
        list(replaceables),
        num_left,
        replace=False
    ))
    subset_right = replaceables - subset_left

    return (
        green_for_subset(s, subset_left, replacement_corpus, replacement_filter),
        green_for_subset(s, subset_right, replacement_corpus, replacement_filter)
    )

# == 3-element generators ==
def gen_3_permutations(s):
    t_size_generator = (lambda x: np.random.randint(2, x + 1) if x >= 2 else 0)
    return [generate_permutation(s, size_generator=t_size_generator) for _ in range(3)]

def gen_3_greenifications(s):
    g_size_generator = (lambda x: np.random.randint(1, x + 1) if x >= 1 else 0)
    return [generate_greenification(s, size_generator=g_size_generator) for _ in range(3)]

# == 3x3 generator ==

def append_randomly(imut, jmut, d1, d2):
    p1 = d1[1:]
    p2 = d2[1:]

    if np.random.random() < 0.5:
        imut.append(p1)
        jmut.append(p2)
    else:
        imut.append(p2)
        jmut.append(p1)

def append_deterministically(imut, jmut, d1, d2):
    p1 = d1[1:]
    p2 = d2[1:]

    imut.append(p1)
    jmut.append(p2)

def make_nxn(pairs, append_random = True):
    if any(p is None for p in pairs) or any(pi is None for p in pairs for pi in p):
        return [IDENTITY] * (len(pairs) + 1), [IDENTITY] * (len(pairs) + 1)

    imut = [IDENTITY]
    jmut = [IDENTITY]

    for pair in pairs:
        if append_random:
            append_randomly(imut, jmut, *pair)
        else:
            append_deterministically(imut, jmut, *pair)

    return imut, jmut

def make_3x3(permutations, greenifications, append_random = True):
    return make_nxn([permutations, greenifications], append_random = append_random)

def make_4x4(randoms, permutations, greenifications):
    return make_nxn([randoms, permutations, greenifications])

def generate_contiguous_3x3(s):
    if len(s) < 6:
        return make_3x3(None, None)

    start, end = sorted(np.random.choice(len(s) - 4, 2))
    end += 4

    return make_3x3(
        generate_contiguous_permutations(s, start_index = start, end_index = end),
        generate_contiguous_greenifications(s, start_index = start, end_index = end)
    )

def generate_3set_3x3(s,
        replacement_corpus=default_replacement_corpus,
        replacement_filter=(lambda x: x in REPLACEABLE_POS)
        ):

    set_choices = np.random.choice(
        3,
        len(s),
        replace = True
    )

    set1 = set(i for i, x in enumerate(set_choices) if x == 1)
    set2 = set(i for i, x in enumerate(set_choices) if x == 2)

    # Two random permutations
    perm1 = permutation_for_subset(set1)
    perm2 = permutation_for_subset(set2)

    # Two random greenifications
    green1 = green_for_subset(s, set1,
            replacement_corpus=replacement_corpus,
            replacement_filter=replacement_filter)
    green2 = green_for_subset(s, set2,
            replacement_corpus=replacement_corpus,
            replacement_filter=replacement_filter)

    return make_3x3([perm1, perm2], [green1, green2], append_random = False)

def generate_arbitrary_3x3(s,
        replacement_corpus=default_replacement_corpus,
        replacement_filter=(lambda x: x in REPLACEABLE_POS)
        ):
    t_size_generator = (lambda x: np.random.randint(2, x + 1) if x >= 2 else 0)
    g_size_generator = (lambda x: np.random.randint(1, x + 1) if x >= 1 else 0)

    # Two random permutations
    perm1 = generate_permutation(s, size_generator=t_size_generator)
    perm2 = generate_permutation(s, size_generator=t_size_generator)

    # Two random greenifications
    green1 = generate_greenification(s,
            size_generator=g_size_generator,
            replacement_corpus=replacement_corpus,
            replacement_filter=replacement_filter)
    green2 = generate_greenification(s,
            size_generator=g_size_generator,
            replacement_corpus=replacement_corpus,
            replacement_filter=replacement_filter)

    return make_3x3([perm1, perm2], [green1, green2])

def generate_generic_4x4(s,
        fourth=generate_single_morph,
        replacement_corpus=default_replacement_corpus,
        replacement_filter=(lambda x: x in REPLACEABLE_POS)):
    t_size_generator = (lambda x: np.random.randint(2, x + 1) if x >= 2 else 0)
    g_size_generator = (lambda x: np.random.randint(1, x + 1) if x >= 1 else 0)

    f1 = fourth(s)
    f2 = fourth(s)

    # Two random permutations
    perm1 = generate_permutation(s, size_generator=t_size_generator)
    perm2 = generate_permutation(s, size_generator=t_size_generator)

    # Two random greenifications
    green1 = generate_greenification(s,
            size_generator=g_size_generator,
            replacement_corpus=replacement_corpus,
            replacement_filter=replacement_filter)
    green2 = generate_greenification(s,
            size_generator=g_size_generator,
            replacement_corpus=replacement_corpus,
            replacement_filter=replacement_filter)

    return make_4x4([f1, f2], [perm1, perm2], [green1, green2])

def generate_morph_4x4(s,
        replacement_corpus=default_replacement_corpus,
        replacement_filter=(lambda x: x in REPLACEABLE_POS)
        ):
    t_size_generator = (lambda x: np.random.randint(2, x + 1) if x >= 2 else 0)
    g_size_generator = (lambda x: np.random.randint(1, x + 1) if x >= 1 else 0)

    morph1 = generate_single_morph(s)
    morph2 = generate_single_morph(s)

    # Two random permutations
    perm1 = generate_permutation(s, size_generator=t_size_generator)
    perm2 = generate_permutation(s, size_generator=t_size_generator)

    # Two random greenifications
    green1 = generate_greenification(s,
            size_generator=g_size_generator,
            replacement_corpus=replacement_corpus,
            replacement_filter=replacement_filter)
    green2 = generate_greenification(s,
            size_generator=g_size_generator,
            replacement_corpus=replacement_corpus,
            replacement_filter=replacement_filter)

    return make_4x4([morph1, morph2], [perm1, perm2], [green1, green2])

def generate_random_4x4(s,
        replacement_corpus=default_replacement_corpus,
        replacement_filter=(lambda x: x in REPLACEABLE_POS)
        ):
    r_size_generator = (lambda x: np.random.randint(1, x + 1) if x >= 1 else 0)
    t_size_generator = (lambda x: np.random.randint(2, x + 1) if x >= 2 else 0)
    g_size_generator = (lambda x: np.random.randint(1, x + 1) if x >= 1 else 0)

    rand1 = generate_random(s, size_generator=r_size_generator)
    rand2 = generate_random(s, size_generator=r_size_generator)

    # Two random permutations
    perm1 = generate_permutation(s, size_generator=t_size_generator)
    perm2 = generate_permutation(s, size_generator=t_size_generator)

    # Two random greenifications
    green1 = generate_greenification(s,
            size_generator=g_size_generator,
            replacement_corpus=replacement_corpus,
            replacement_filter=replacement_filter)
    green2 = generate_greenification(s,
            size_generator=g_size_generator,
            replacement_corpus=replacement_corpus,
            replacement_filter=replacement_filter)

    return make_4x4([rand1, rand2], [perm1, perm2], [green1, green2])

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

def tokenize_matrix(prop, tokenizer = None):
    if tokenizer is None:
        tokenizer = get_gpt2_tokenizer('en')

    def run_tokenization(d):
        result = []
        for arec, brec, matrix in tqdm(d[prop]):
            result.append([
                [tokenizer(s)['input_ids'] for s in row]
                for row in matrix
            ])
        return result
    return run_tokenization

def tokenized_prop_for(m_prop, tokenizer = None, tokenizer_label = 'tok'):
    if tokenizer is None:
        tokenizer = get_gpt2_tokenizer('en')

    return Property(
        [m_prop],
        m_prop.name + '_%s' % tokenizer_label,
        tokenize_matrix(m_prop, tokenizer = tokenizer)
    )

criterion = torch.nn.CrossEntropyLoss(reduction='sum')

def get_gpt2_pipeline(lang):
    score = gpt2_scorer(get_gpt2_lm(lang))
    tokenizer = get_gpt2_tokenizer(lang)

    return lambda s: score(tokenizer(s).input_ids)

def gpt2_scorer(lmmodel):
    def score(t):
        with torch.no_grad():
            t = torch.LongTensor(t).unsqueeze(0).cuda()
            result = lmmodel(t).logits[0]
            return criterion(result[:-1], t[0][1:]).cpu().numpy().tolist()
    return score

def lstm_scorer(model):
    model.eval()
    def scorer(t):
        if len(t) <= 1:
            return 0
        with torch.no_grad():
            t = torch.LongTensor(t)
            inp = torch.nn.utils.rnn.pack_sequence([t[:-1]]).cuda()
            targ = torch.nn.utils.rnn.pack_sequence([t[1:]]).cuda()

            result = model(inp, model.init_state(1, cuda = True))[0]
            return criterion(result.data, targ.data).cpu().numpy().tolist()
    return scorer

def score_matrix(prop, model_fn = None):
    if model_fn is None:
        model_fn = gpt2_scorer(get_gpt2_lm('en'))

    def run_scoring(d):
        result = []
        for matrix in tqdm(d[prop]):
            result.append([
                [model_fn(tokens) if 0 < len(tokens) < 1024 else None for tokens in row]
                for row in matrix
            ])
        return result
    return run_scoring

def scored_prop_for(m_prop, model_fn = None, tokenizer = None,
        model_label = 'scored',
        tokenizer_label = 'tok'):
    t_prop = tokenized_prop_for(m_prop,
            tokenizer = tokenizer, tokenizer_label = tokenizer_label)

    return Property(
        [t_prop],
        m_prop.name + '_%s' % model_label,
        score_matrix(t_prop, model_fn = model_fn),
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

def get_mse(model, dataset, i, j, clip_outliers = None):
    BATCH_SIZE = 128

    if clip_outliers is not None:
        dataset = [x for x in dataset if not any(k is None or abs(k) > clip_outliers for row in x for k in row)]
    else:
        dataset = [x for x in dataset if not any(k is None for row in x for k in row)]

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

    criterion = torch.nn.MSELoss(reduction='sum')

    total_loss = 0
    total_examples = 0

    with torch.no_grad():
        for r, a, b, d in batches:
            # Inputs a:
            prediction = model(r.cuda(), a.cuda(), b.cuda())
            loss = criterion(prediction, d.cuda())

            total_loss += loss
            total_examples += len(r)

    return total_loss / total_examples

def entanglement_model(s_prop, i, j, model_factory, model_name,
        epochs = 100, lr=1e-1):

    BATCH_SIZE = 128
    model = model_factory().cuda()

    def train(d):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        dataset = [x for x in d[s_prop] if not any(k is None for row in x for k in row)]

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
        '%s_%d_%d_model_%s_%d_%f' % (s_prop.name, i, j, model_name, epochs, lr),
        train,
        version = 2
    )

JUST_MEAN = lambda x: torch.mean(x, keepdim=True)

def first_moments(n):
    return lambda x: torch.mean(
        torch.pow(x.unsqueeze(1), torch.arange(n).unsqueeze(0)),
        dim = 1
    )

class DistributionalPnorm(torch.nn.Module):
    def __init__(self, p = 1, chars = JUST_MEAN):
        super(DistributionalPnorm, self).__init__()
        self.p = torch.nn.Parameter(torch.Tensor([p]))

    def forward(self, dist_a, dist_b):
        # Each of dist_a and dist_b are allowed to have
        # multiple columns
        sign_a = torch.sign(dist_a)
        dist_a *= sign_a
        a_lse, a_signs = signed_logsumexp(
            torch.log(dist_a + t_eps) * self.p,
            sign_a
        )
        a_shaped, _ = torch.sort(a_lse * a_signs)
        #a_pmean = torch.mean(a_lse * a_signs)

        sign_b = torch.sign(dist_b)
        dist_b *= sign_b
        b_lse, b_signs = signed_logsumexp(
            torch.log(dist_b + t_eps) * self.p,
            sign_b
        )
        b_shaped, _ = torch.sort(b_lse * b_signs)
        #b_pmean = torch.mean(b_lse * b_signs)

        return a_shaped, b_shaped #chars, b_chars

def prepare_dists(
        before_root, before_dist,
        after_root, after_dist):

    before = torch.stack([
        (after_root - before_root).expand_as(before_dist),
        (before_dist - before_root)
    ], dim = 1)
    after = (after_dist - before_root).unsqueeze(1)

    return before, after

def solve_distributional_pnorm(
        before_root, before_dist,
        after_root, after_dist,
        epochs=3000, log_every=50):
    before, after = prepare_dists(
        before_root, before_dist,
        after_root, after_dist
    )
    print('SOLVING WITH DISTRIBUTIONS')
    print(before)
    print(after)
    model = DistributionalPnorm().cuda()
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        optimizer.zero_grad()
        a_pred, b_pred = model(before, after)
        loss = criterion(a_pred, b_pred)
        if epoch % log_every == 0:
            print('Iter %d with p=%f giving loss %f' % (epoch, model.p, loss))
        loss.backward()
        optimizer.step()

    return model.p

def measure_entanglement_with(
        mutator, before, after, generator, samples, score_fn, tokenizer):

    before_root = torch.Tensor([score_fn(tokenizer(mutator.deparse(before))['input_ids'])]).cuda()
    before_dist = torch.Tensor([
        score_fn(
            tokenizer(mutator.deparse(generator(before)[2](before)))['input_ids']
        )
        for _ in trange(samples)
    ]).cuda()

    after_root = torch.Tensor([score_fn(tokenizer(mutator.deparse(after))['input_ids'])]).cuda()
    after_dist = torch.Tensor([
        score_fn(
            tokenizer(mutator.deparse(generator(after)[2](after)))['input_ids']
        )
        for _ in trange(samples)
    ]).cuda()

    return solve_distributional_pnorm(
        before_root, before_dist,
        after_root, after_dist)

def measure_type(
        before, after,
        syntax_generator = generate_permutation,
        semantics_generator = generate_greenification,
        samples = 1024,
        score_fn = None,
        tokenizer = None,
        language = ENGLISH_NLTK):
    if score_fn is None:
        score_fn = gpt2_scorer(get_gpt2_lm('en'))
    if tokenizer is None:
        tokenizer = get_gpt2_tokenizer('en')

    mutator = Mutator(language)
    before, after = mutator.parse(before), mutator.parse(after)

    syntax_entanglement = measure_entanglement_with(
        mutator, before, after, syntax_generator, samples, score_fn, tokenizer
    )
    semantics_entanglement = measure_entanglement_with(
        mutator, before, after, semantics_generator, samples, score_fn, tokenizer
    )

    return syntax_entanglement, semantics_entanglement

def four_score_prop(m_prop,
        model_fn = None,
        model_label = 'scored',
        tokenizer = None,
        tokenizer_label = 'tok',
        score_type = 'abs'):
    scoring_function = ({
        'abs': abs_score_for,
        'nuc': nuc_score_for,
        'mut': mut_score_for
    })[score_type]
    s_prop = scored_prop_for(m_prop,
        model_fn = model_fn,
        model_label = model_label,
        tokenizer = tokenizer,
        tokenizer_label = tokenizer_label
    )

    return Property(
        [s_prop],
        '%s_%s_%s_evals' % (m_prop.name, model_label, score_type),
        (lambda d: four_minor_scores(d, s_prop, f = scoring_function)),
        version = 1
    )
