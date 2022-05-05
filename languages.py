import nltk
#from stanfordcorenlp import StanfordCoreNLP
import stanza
import unidecode

# English
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
nltk_tokenizer = TreebankWordTokenizer()
nltk_detokenizer = TreebankWordDetokenizer()

# French, Chinese
from sacremoses import MosesTokenizer, MosesDetokenizer
CORENLP_PATH = '/raid/lingo/abau/scratch/stanford-corenlp-4.4.0'

class LanguageToolset:
    def __init__(self,
            tokenize = (lambda x: nltk_tokenizer.tokenize(unidecode.unidecode(x))),
            detokenize = (lambda x: nltk_detokenizer.detokenize(x)),
            pos_tag = (lambda x: nltk.pos_tag(x))):
        self.tokenize = tokenize
        self.pos_tag = pos_tag
        self.detokenize = detokenize

ENGLISH_NLTK = LanguageToolset()

def unify(pos, feats):
    return pos + '::' + ('|'.join(sorted(feats.split('|'))) if feats is not None else '_')

class StanfordMosesToolset(LanguageToolset):
    def __init__(self, lang):
        self.lang = lang

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def detokenize(self, t):
        return self.detokenizer.detokenize(t)

    def pos_tag(self, s):
        annotated = self.nlp([s])
        return [
            (word.text, unify(word.pos, word.feats))
            for word in annotated.sentences[0].words
        ]

    def __enter__(self):
        #stanza.download(self.lang)
        if self.lang == 'zh':
            self.nlp = stanza.Pipeline(self.lang, processors='tokenize,pos', tokenize_pretokenized=True, tokenize_no_ssplit=True)
        else:
            self.nlp = stanza.Pipeline(self.lang, processors='tokenize,mwt,pos', tokenize_pretokenized=True, tokenize_no_ssplit=True)
        self.tokenizer = MosesTokenizer(lang=self.lang)
        self.detokenizer = MosesDetokenizer(lang=self.lang)

        '''
        self.corenlp = StanfordCoreNLP(CORENLP_PATH, lang=self.lang)
        '''

        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass

def make_filter_for_rp_list(rp, corpus):
    cache = {}
    def check(pos):
        if pos not in cache:
            cache[pos] = pos.split('::')[0] in rp and pos in corpus and sum(1 for x in corpus[pos] if corpus[pos][x] > 0) >= 2
        return cache[pos]
    return check
