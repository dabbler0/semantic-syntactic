from tools import *
from matrices_modular import *

def parse(s):
    return {
        i: (i, tok, pos) for i, (tok, pos) in enumerate(
            nltk.pos_tag(nltk_tokenizer.tokenize(unidecode.unidecode(s)))
        )
    }

def markup(f):
    if len(f) == 3:
        return f[1]
    else:
        starts = ''
        ends = ''
        if 'permuted' in f[3]:
            starts += '<span style="border: 1px solid black">'
            ends  = '</span>' + ends
        if 'greenified' in f[3]:
            starts += '<span style="background-color: yellow">'
            ends += '</span>' + ends
        return starts + f[1] + ends

def markup_deparse(s):
    inverted = {
        s[i][0]: s[i]
        for i in s
    }
    flattened = [inverted[i] for i in range(len(inverted))]
    return ' '.join(markup(f) for f in flattened)

class Visualizer:
    def __init__(self, m_prop, s_prop):
        self.m_prop = m_prop
        self.s_prop = s_prop
        self.score_prop = scored_prop_for(m_prop)

    def visualize_matrix(self, s):
        arecs, brecs, matrix = s[self.m_prop]
        scores = s[self.score_prop]
        original = parse(matrix[0][0])

        result = '<table>'
        for i, arec in enumerate(arecs):
            result += '<tr>'
            for j, brec in enumerate(brecs):
                result += '<td>'
                result += '<b>%f: </b>' % scores[i][j]
                result += markup_deparse(
                    apply_record(
                        apply_record(
                            original, brec, markup=True
                        ),
                        arec, markup=True
                    )
                )
                result += '</td>'
            result += '</tr>'
        result += '</table>'
        return result

        # TODO modify visualizer to show changes
        '''
        result = '<table>'

        for row in matrix:
            result += '<tr>'
            for cell in row:
                result += cell

        return result
        '''
