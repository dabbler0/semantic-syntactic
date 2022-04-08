from matrices_modular import *

# Set up mutator
def t_size_generator(l):
    return min(l, np.random.randint(2, 16))

def g_size_generator(l):
    return min(l, np.random.randint(1, 16))

mutator = GreenOrderMutator(
    lambda s: generate_disjoint_3x3(s, t_size_generator, g_size_generator)
)

m_prop = MatrixProp(sentence_prop, 'disjoint_broad_3x3', mutator)
abs_score_prop = four_score_prop(m_prop)

tiny = Dataset('tiny-0')
scores = tiny[abs_score_prop]

# Contiguous
'''
contig_mutator = GreenOrderMutator(
    lambda s: generate_contiguous_3x3(s)
)

m_prop = MatrixProp(sentence_prop, 'big_contiguous_3x3', contig_mutator)
abs_score_prop = four_score_prop(m_prop)

tiny = Dataset('tiny-0')
scores = tiny[abs_score_prop]
'''

'''
contig_mutator = GreenOrderMutator(
    lambda s: generate_contiguous_3x3(s)
)

m_prop = MatrixProp(sentence_prop, 'big_nonid_contiguous_3x3', contig_mutator)
abs_score_prop = four_score_prop(m_prop)
mut_score_prop = four_score_prop(m_prop, score_type='mut')

tiny = Dataset('tiny-0')
scores = tiny[abs_score_prop]
mut_scores = tiny[mut_score_prop]
'''

distant_mutator = GreenOrderMutator(
    lambda s: generate_distant_3x3(s)
)

m_prop = MatrixProp(sentence_prop, 'big_nonid_distant_3x3', distant_mutator)
abs_score_prop = four_score_prop(m_prop)
mut_score_prop = four_score_prop(m_prop, score_type='mut')

tiny = Dataset('tiny-0')
scores = tiny[abs_score_prop]
mut_scores = tiny[mut_score_prop]

'''
from matrices import *

# Ordinary scoring
tiny = Dataset('tiny-0')
four_scores = tiny[gpt_four_score_prop]

reference = Dataset('tiny-1')
ngram_reference_prop = ngram_3x3_prop(reference)
ngram_abs_scores = tiny[four_score_prop(ngram_reference_prop)]
'''
