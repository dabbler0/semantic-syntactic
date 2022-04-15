import nltk
from stanfordcorenlp import StanfordCoreNLP

# English
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
nltk_tokenizer = TreebankWordTokenizer()
nltk_detokenizer = TreebankWordDetokenizer()

# French, Chinese
from sacremoses import MosesTokenizer, MosesDetokenizer
CORENLP_PATH = '/raid/lingo/abau/scratch/stanford-corenlp-4.4.0'

class LanguageToolset:
    def __init__(self,
            tokenize = (lambda x: nltk_tokenizer.tokenize(x)),
            detokenize = (lambda x: nltk_detokenizer.detokenize(x)),
            pos_tag = (lambda x: nltk.pos_tag(x))):
        self.tokenize = tokenize
        self.pos_tag = pos_tag
        self.detokenize = detokenize

ENGLISH_NLTK = LanguageToolset()

class StanfordMosesToolset(LanguageToolset):
    def __init__(self, lang):
        self.lang = lang

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def detokenize(self, t):
        return self.detokenizer.detokenize(t)

    def pos_tag(self, s):
        return self.corenlp.annotate(s)

    def __enter__(self):
        self.corenlp = StanfordCoreNLP(CORENLP_PATH, lang=self.lang)
        self.tokenizer = MosesTokenizer(lang=self.lang)
        self.detokenizer = MosesDetokenizer(lang=self.lang)

    def __exit__(self):
        self.corenlp.close()
