from matrices_modular import *
from entanglement_models import *
from lstms import *

current_experiment = 'TRIADS'

if current_experiment == 'TRIADS':
    tiny = Dataset('tiny-0')

    # Perm
    perm_mutator = AllSubsetsMutator(
        gen_3_permutations
    )

    p_m_prop = MatrixProp(sentence_prop, 'perm_subsets', perm_mutator, version = 1)
    p_score_prop = scored_prop_for(p_m_prop)
    p_scores = tiny[p_score_prop]

    # Green
    green_mutator = AllSubsetsMutator(
        gen_3_greenifications
    )

    g_m_prop = MatrixProp(sentence_prop, 'green_subsets', green_mutator, version = 1)
    g_score_prop = scored_prop_for(g_m_prop)
    g_scores = tiny[g_score_prop]

if current_experiment == 'SOLVE_S':
    mutator = GreenOrderMutator(
        lambda s: generate_disjoint_3x3(s, t_size_generator, g_size_generator)
    )

    m_prop = MatrixProp(sentence_prop, 'disjoint_broad_3x3', mutator)
    s_prop = scored_prop_for(m_prop)

    tt_model_prop = entanglement_model(
        s_prop, 1, 1,
        (lambda: SelfEntanglementModel()),
        'self1param'
    )
    mm_model_prop = entanglement_model(
        s_prop, 2, 2,
        (lambda: SelfEntanglementModel()),
        'self1param'
    )

    tiny = Dataset('tiny-0')
    tt_trained_model = tiny[tt_model_prop]
    mm_trained_model = tiny[mm_model_prop]

if current_experiment == 'LSTM':
    contig_mutator = GreenOrderMutator(
        lambda s: generate_contiguous_3x3(s)
    )

    m_prop = MatrixProp(sentence_prop, 'big_nonid_contiguous_3x3', contig_mutator)

    tiny = Dataset('tiny-0')
    large = Dataset('large-1')
    model_prop = lstm_prop(sentence_prop)
    lstm_model = large[model_prop]

    abs_score_prop = four_score_prop(m_prop,
        model_fn = lstm_scorer(lstm_model),
        model_label = 'lstmscored'
    )
    mut_score_prop = four_score_prop(m_prop,
        model_fn = lstm_scorer(lstm_model),
        model_label = 'lstmscored',
        score_type='mut'
    )

    scores = tiny[abs_score_prop]
    mut_scores = tiny[mut_score_prop]

# Arbitrary
if current_experiment == 'ARBITRARY':
    mutator = GreenOrderMutator(
        generate_arbitrary_3x3
    )

    m_prop = MatrixProp(sentence_prop, 'arbitrary_3x3', mutator)
    abs_score_prop = four_score_prop(m_prop)

    tiny = Dataset('tiny-0')
    scores = tiny[abs_score_prop]

# Disjoint broad
if current_experiment == 'DISJOINT_BROAD':
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
if current_experiment == 'CONTIGUOUS':
    contig_mutator = GreenOrderMutator(
        lambda s: generate_contiguous_3x3(s)
    )

    m_prop = MatrixProp(sentence_prop, 'big_nonid_contiguous_3x3', contig_mutator)
    abs_score_prop = four_score_prop(m_prop)
    mut_score_prop = four_score_prop(m_prop, score_type='mut')

    tiny = Dataset('tiny-0')
    scores = tiny[abs_score_prop]
    mut_scores = tiny[mut_score_prop]

# Distant
if current_experiment == 'DISTANT':
    distant_mutator = GreenOrderMutator(
        lambda s: generate_distant_3x3(s)
    )

    m_prop = MatrixProp(sentence_prop, 'big_nonid_distant_3x3', distant_mutator)
    abs_score_prop = four_score_prop(m_prop)
    mut_score_prop = four_score_prop(m_prop, score_type='mut')

    tiny = Dataset('tiny-0')
    scores = tiny[abs_score_prop]
    mut_scores = tiny[mut_score_prop]
