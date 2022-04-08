#!/bin/env python

import nltk
import math
import itertools

class NgramModel(object):
    def __init__(self, train_data, n = 3, vocab_size = 50257, smoothing_config = { 'type': 'laplace-flat', 'l': 1 }):
        self.n = n
        self.tokens = train_data
        self.vocab_size = vocab_size
        self.total_len = sum(len(x) for x in self.tokens)
        self.smoothing_config = smoothing_config
        self.ngram_dists = {
            k: nltk.FreqDist(
                itertools.chain(*(nltk.ngrams(t, k) for t in self.tokens))
            ) for k in range(self.n + 1)
        }
        self._smooth()

    def _smooth(self):
        if self.smoothing_config['type'] == 'laplace-flat':
            self.smoothed_probs = {}
            self.discount_probs = {}

            laplace = self.smoothing_config['l']

            for k in range(self.n + 1):
                self.smoothed_probs[k] = {}
                self.discount_probs[k] = {}
                for ngram in self.ngram_dists[k]:
                    if k >= 1:
                        self.smoothed_probs[k][ngram] = math.log((
                            self.ngram_dists[k][ngram] + laplace
                        ) / (
                            self.ngram_dists[k - 1][ngram[:-1]] + laplace * self.vocab_size
                        ))

                    if k < self.n:
                        self.discount_probs[k][ngram] = math.log(laplace / (self.ngram_dists[k][ngram] + laplace * self.vocab_size))

            self.discount_probs[0][()] = math.log(laplace / (self.total_len + laplace * self.vocab_size))

    def get_prob(self, ngram):
        ngram = tuple(ngram)
        if self.smoothing_config['type'] == 'laplace-flat':
            if ngram in self.smoothed_probs[len(ngram)]:
                return self.smoothed_probs[len(ngram)][ngram]
            elif ngram[:-1] in self.discount_probs[len(ngram) - 1]:
                return self.discount_probs[len(ngram) - 1][ngram[:-1]]
            else:
                return self.get_prob(ngram[:-1])

    def score(self, sentence):
        if self.smoothing_config['type'] == 'laplace-flat':
            return 1 / len(sentence) * (
                sum(self.get_prob(ngram) for ngram in nltk.ngrams(sentence, self.n))
                + sum(self.get_prob(sentence[:k]) for k in range(1, self.n))
            )
