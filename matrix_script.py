from matrices_modular import *
from entanglement_models import *
from lstms import *
import sys

current_experiment = sys.argv[1]

print('Running experiment', current_experiment)

if current_experiment == 'ZH':
    with StanfordMosesToolset('zh') as CHINESE:
        sentence_prop = InjectedProperty('sentence')

        zh_tiny1 = Dataset('zh-tiny-1')
        zh_tiny0 = Dataset('zh-tiny-0')

        zh_tc_prop = Property(
            [sentence_prop],
            'tagcorpus-zh3',
            lambda d: get_tag_corpus(d, language=CHINESE)
        )

        replacement_corpus = zh_tiny1[zh_tc_prop]

        ZH_REPLACEABLES = ('ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB')

        mutator = GreenOrderMutator(
            (lambda s:
                generate_arbitrary_3x3(
                    s,
                    replacement_corpus = replacement_corpus,
                    replacement_filter = make_filter_for_rp_list(ZH_REPLACEABLES, replacement_corpus)
                )
            ),
            language = CHINESE
        )

        m_prop = MatrixProp(sentence_prop, 'arbitrary_3x3_3', mutator)

        mutated_sentences = zh_tiny0[m_prop]
        abs_score_prop = four_score_prop(m_prop,
            model_fn = gpt2_scorer(get_zh_lm()),
            tokenizer = get_zh_tokenizer(),
            tokenizer_label = 'zhtok',
            model_label = 'zhscored'
        )

        scored = zh_tiny0[abs_score_prop]

if current_experiment == 'FI':
    with StanfordMosesToolset('fi') as FINNISH:
        sentence_prop = InjectedProperty('sentence')

        fi_tiny1 = Dataset('fi-tiny-1')
        fi_tiny0 = Dataset('fi-tiny-0')

        fi_tc_prop = Property(
            [sentence_prop],
            'tagcorpus-fi',
            lambda d: get_tag_corpus(d, language=FINNISH)
        )

        replacement_corpus = fi_tiny1[fi_tc_prop]

        FI_REPLACEABLES = ('ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB')

        mutator = GreenOrderMutator(
            (lambda s:
                generate_arbitrary_3x3(
                    s,
                    replacement_corpus = replacement_corpus,
                    replacement_filter = make_filter_for_rp_list(FI_REPLACEABLES, replacement_corpus)
                )
            ),
            language = FINNISH
        )

        m_prop = MatrixProp(sentence_prop, 'arbitrary_3x3_3', mutator)

        mutated_sentences = fi_tiny0[m_prop]
        abs_score_prop = four_score_prop(m_prop,
            model_fn = gpt2_scorer(get_gpt2_lm('fi')),
            tokenizer = get_gpt2_tokenizer('fi'),
            tokenizer_label = 'fitok',
            model_label = 'fiscored'
        )

        scored = fi_tiny0[abs_score_prop]

if current_experiment == 'EN_STANZA_LSTM':
    with StanfordMosesToolset('en') as ENGLISH:
        sentence_prop = InjectedProperty('sentence')

        en_tiny1 = Dataset('tiny-1')
        en_tiny0 = Dataset('tiny-0')

        en_tc_prop = Property(
            [sentence_prop],
            'tagcorpus-stanza',
            lambda d: get_tag_corpus(d, language=ENGLISH)
        )

        replacement_corpus = en_tiny1[en_tc_prop]

        ENGLISH_REPLACEABLES = ('ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB')

        mutator = GreenOrderMutator(
            (lambda s:
                generate_arbitrary_3x3(
                    s,
                    replacement_corpus = replacement_corpus,
                    replacement_filter = make_filter_for_rp_list(ENGLISH_REPLACEABLES, replacement_corpus)
                )
            ),
            language = ENGLISH
        )

        m_prop = MatrixProp(sentence_prop, 'arbitrary_3x3_stanza', mutator)

        large = Dataset('large-1')
        model_prop = lstm_prop(sentence_prop)
        lstm_model = large[model_prop].cuda()

        mutated_sentences = en_tiny0[m_prop]
        abs_score_prop = four_score_prop(m_prop,
            model_fn = lstm_scorer(lstm_model),
            tokenizer = get_gpt2_tokenizer('en'),
            tokenizer_label = 'tok',
            model_label = 'lstmscored'
        )

        scored = en_tiny0[abs_score_prop]

if current_experiment == 'EN_STANZA_DISJOINT':
    with StanfordMosesToolset('en') as ENGLISH:
        sentence_prop = InjectedProperty('sentence')

        en_tiny1 = Dataset('tiny-1')
        en_tiny0 = Dataset('tiny-0')

        en_tc_prop = Property(
            [sentence_prop],
            'tagcorpus-stanza',
            lambda d: get_tag_corpus(d, language=ENGLISH)
        )

        replacement_corpus = en_tiny1[en_tc_prop]

        ENGLISH_REPLACEABLES = ('ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB')

        mutator = GreenOrderMutator(
            (lambda s:
                generate_3set_3x3(
                    s,
                    replacement_corpus = replacement_corpus,
                    replacement_filter = make_filter_for_rp_list(ENGLISH_REPLACEABLES, replacement_corpus)
                )
            ),
            language = ENGLISH
        )

        m_prop = MatrixProp(sentence_prop, 'disjoint_3x3_stanza', mutator)

        mutated_sentences = en_tiny0[m_prop]
        abs_score_prop = four_score_prop(m_prop,
            model_fn = gpt2_scorer(get_gpt2_lm('en')),
            tokenizer = get_gpt2_tokenizer('en'),
            tokenizer_label = 'tok',
            model_label = 'scored'
        )

        scored = en_tiny0[abs_score_prop]

if current_experiment == 'EN_STANZA':
    with StanfordMosesToolset('en') as ENGLISH:
        sentence_prop = InjectedProperty('sentence')

        en_tiny1 = Dataset('tiny-1')
        en_tiny0 = Dataset('tiny-0')

        en_tc_prop = Property(
            [sentence_prop],
            'tagcorpus-stanza',
            lambda d: get_tag_corpus(d, language=ENGLISH)
        )

        replacement_corpus = en_tiny1[en_tc_prop]

        ENGLISH_REPLACEABLES = ('ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB')

        mutator = GreenOrderMutator(
            (lambda s:
                generate_arbitrary_3x3(
                    s,
                    replacement_corpus = replacement_corpus,
                    replacement_filter = make_filter_for_rp_list(ENGLISH_REPLACEABLES, replacement_corpus)
                )
            ),
            language = ENGLISH
        )

        m_prop = MatrixProp(sentence_prop, 'arbitrary_3x3_stanza', mutator)

        mutated_sentences = en_tiny0[m_prop]
        abs_score_prop = four_score_prop(m_prop,
            model_fn = gpt2_scorer(get_gpt2_lm('en')),
            tokenizer = get_gpt2_tokenizer('en'),
            tokenizer_label = 'tok',
            model_label = 'scored'
        )

        scored = en_tiny0[abs_score_prop]

if current_experiment == 'FR':
    with StanfordMosesToolset('fr') as FRENCH:
        sentence_prop = InjectedProperty('sentence')

        fr_tiny1 = Dataset('fr-tiny-1')
        fr_tiny0 = Dataset('fr-tiny-0')

        fr_tc_prop = Property(
            [sentence_prop],
            'tagcorpus-fr3',
            lambda d: get_tag_corpus(d, language=FRENCH)
        )

        replacement_corpus = fr_tiny1[fr_tc_prop]

        FR_REPLACEABLES = ('ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB')

        mutator = GreenOrderMutator(
            (lambda s:
                generate_arbitrary_3x3(
                    s,
                    replacement_corpus = replacement_corpus,
                    replacement_filter = make_filter_for_rp_list(FR_REPLACEABLES, replacement_corpus)
                )
            ),
            language = FRENCH
        )

        m_prop = MatrixProp(sentence_prop, 'arbitrary_3x3_3', mutator)

        mutated_sentences = fr_tiny0[m_prop]
        abs_score_prop = four_score_prop(m_prop,
            model_fn = gpt2_scorer(get_gpt2_lm('fr')),
            tokenizer = get_gpt2_tokenizer('fr'),
            tokenizer_label = 'frtok',
            model_label = 'frscored'
        )

        scored = fr_tiny0[abs_score_prop]

if current_experiment == 'TRIADS':
    tiny = Dataset('tiny-0')

    # Perm
    perm_mutator = AllSubsetsMutator(
        gen_3_permutations
    )

    p_m_prop = MatrixProp(sentence_prop, 'perm_subsets_v1', perm_mutator)
    p_score_prop = scored_prop_for(p_m_prop)
    p_scores = tiny[p_score_prop]

    # Green
    green_mutator = AllSubsetsMutator(
        gen_3_greenifications
    )

    g_m_prop = MatrixProp(sentence_prop, 'green_subsets_v1', green_mutator)
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
        generate_arbitrary_3x3
    )

    m_prop = MatrixProp(sentence_prop, 'arbitrary_3x3', contig_mutator)

    tiny = Dataset('tiny-0')
    large = Dataset('large-1')
    model_prop = lstm_prop(sentence_prop)
    lstm_model = large[model_prop].cuda()

    abs_score_prop = four_score_prop(m_prop,
        model_fn = lstm_scorer(lstm_model),
        model_label = 'lstmscored2'
    )
    mut_score_prop = four_score_prop(m_prop,
        model_fn = lstm_scorer(lstm_model),
        model_label = 'lstmscored2',
        score_type='mut'
    )

    scores = tiny[abs_score_prop]
    mut_scores = tiny[mut_score_prop]

if current_experiment == 'DROP-DUP':
    for mut_type in ('drop', 'dup'):
        for tag_start in ('J', 'R', 'N', 'V'):
            filter_fn = (lambda x: x[0] == tag_start)

            size_generator = (lambda x:
                    np.random.randint(1, x + 1) if x > 1 else 0)

            generator = (generate_duplications if mut_type == 'drop' else
                    generate_drops)

            mutator = GreenOrderMutator(
                lambda s: generate_generic_4x4(
                    s,
                    fourth = lambda s1: generator(
                        s1,
                        pos_filter = filter_fn,
                        size_generator = size_generator
                    )
                )
            )

            m_prop = MatrixProp(sentence_prop,
                    '%s_%s_4x4' % (mut_type, tag_start), mutator)
            score_prop = scored_prop_for(m_prop)

            tiny = Dataset('tiny-0')
            scores = tiny[score_prop]

if current_experiment == 'MORPH':
    mutator = GreenOrderMutator(
        generate_morph_4x4
    )

    m_prop = MatrixProp(sentence_prop, 'morph_4x4', mutator)
    score_prop = scored_prop_for(m_prop)

    tiny = Dataset('tiny-0')
    scores = tiny[score_prop]

if current_experiment == 'RANDOM':
    mutator = GreenOrderMutator(
        generate_random_4x4
    )

    m_prop = MatrixProp(sentence_prop, 'random_4x4', mutator)
    score_prop = scored_prop_for(m_prop)

    tiny = Dataset('tiny-0')
    scores = tiny[score_prop]

# Arbitrary, with new greenifications
if current_experiment == 'ARBITRARY':
    mutator = GreenOrderMutator(
        generate_arbitrary_3x3
    )

    m_prop = MatrixProp(sentence_prop, 'arbitrary_3x3', mutator)
    abs_score_prop = four_score_prop(m_prop)

    tiny = Dataset('tiny-0')
    scores = tiny[abs_score_prop]

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
