import json
import os

class SyntaxGymInfo:
    def __init__(self):
        '''
        if os.path.exists('syntaxgym-config.json'):
            with open('syntaxgym-config.json') as f:
                self.names, self.criteria = json.load(f)
        else:
        '''
        self.names = {}
        self.criteria = {}

    def register(self, number, indicator, name, criterion):
        self.names[number, indicator] = name
        self.criteria[number, indicator] = criterion
        #self.save()

I = SyntaxGymInfo()

def register_center_embedding(n, label):
    I.register(269, 0,
        ('Center embedding',
         label,
         'Plausibility'),
        lambda x: x['implaus'] - x['plaus']
    )

register_center_embedding(269, 'No modifier')

def register_garden_path(n, label,
        good_ambig = 'unreduced_ambig',
        good_unambig = 'unreduced_unambig',
        bad_ambig = 'reduced_ambig',
        bad_unambig = 'reduced_unambig'):
    I.register(
        n, 0,
        ('Garden path',
         label,
         'Ambiguity makes resolution surprising'),
        lambda x: (
            (x[bad_ambig] - x[good_ambig]) -
            (x[bad_unambig] - x[good_unambig])
        )
    )
    I.register(n, 1,
        ('Garden path',
         label,
         'Ambiguity hurts'),
        lambda x: (x[bad_ambig] - x[bad_unambig])
    )
    I.register(n, 2,
        ('Garden path',
         label,
         'Prefers natural version'),
        lambda x: (x[bad_ambig] - x[good_ambig])
    )

register_garden_path(265, 'Main verb')

def register_gap(n, label):
    I.register(267, 0,
        ('Filler gap dependencies',
         label,
         'Gap requires "what"'),
        lambda x: (x['that_gap'] - x['what_gap'])
    )
    I.register(267, 1,
        ('Filler gap dependencies',
         label,
         'No-gap requires "that"'),
       lambda x: (x['what_no-gap'] - x['that_no-gap'])
    )

register_gap(267, '4 sentential embeddings')

def register_number(n, label):
    I.register(n, 0, 
        ('Number agreement',
         label,
         'Plural'),
        lambda x: (x['mismatch_plural'] - x['match_plural'])
    )
    I.register(n, 1,
        ('Number agreement',
         label,
         'Singular'),
        lambda x: (x['mismatch_sing'] - x['match_sing'])
    )

register_number(260, 'Masculine reflexive, subject relative clause')

register_garden_path(253, 'NP/Z',
    good_ambig = 'ambig_comma',
    bad_ambig = 'ambig_nocomma',
    good_unambig = 'unambig_comma',
    bad_unambig = 'unambig_nocomma')

def register_npl(n, label):
    I.register(n, 0,
        ('Negative polarity licensing',
         label,
         'Must lead with neg when nested is neg'),
        lambda x: (x['pos_neg'] - x['neg_neg'])
    )
    I.register(n, 1,
        ('Negative polarity licensing',
         label,
         'Neg must come before pos'),
        lambda x: (x['pos_neg'] - x['neg_pos'])
    )
    I.register(n, 2,
        ('Negative polarity licensing',
         label,
         'Must lead with neg when nested is pos'),
        lambda x: (x['pos_pos'] - x['neg_pos'])
    )

register_npl(258, '"ever", object relative clause')

register_number(248, 'Masculine reflexive, prepositional phrase')

register_number(261, 'Verb, subject relative clause')

register_gap(254, 'Hierarchy')

register_npl(255, 'Any, object relative clause')

register_number(252, 'Feminine reflexive, prepositional phrase')

register_npl(256, 'Any, subject relative clause')

register_center_embedding(249, 'With modifier')

register_number(259, 'Verb, prepositional phrase')

def register_subordination(n, label):
    I.register(n, 0,
        ('Subordination',
         label,
         'Matrix requires subordination'),
        lambda x: x['no-sub_matrix'] - x['sub_matrix'])
    I.register(n, 1,
        ('Subordination',
         label,
         'Subordination requires matrix'),
        lambda x: x['sub_no-matrix'] - x['no-sub_no-matrix'])

register_subordination(263, 'Plain')

register_garden_path(251, 'NP/Z Overt Object',
    bad_ambig = 'no-obj_no-comma',
    good_ambig = 'no-obj_comma',
    bad_unambig = 'obj_no-comma',
    good_unambig = 'obj_comma')

register_npl(257, 'Ever, subject relative clause')

def register_cleft(n, label):
    I.register(n, 0,
        ('Cleft structure',
         label,
         'Noun phrases'),
        lambda x: (x['np_mismatch'] - x['np_match']))
    I.register(n, 1,
        ('Cleft structure',
         label,
         'Verb phrases'),
        lambda x: (x['vp_mismatch'] - x['vp_match']))

register_cleft(247, 'Plain')

register_subordination(250, 'Object relative clause')

register_gap(266, 'Extraction from prepositional phrase')

register_subordination(268, 'Preopositional phrase')

register_cleft(262, 'With modifier')

register_number(264, 'Feminine reflexive, with object relative clause')

SYNTAX_GYM_INFO = I
