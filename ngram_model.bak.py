import torch
import numpy as np
from tqdm import tqdm, trange
#import simple_good_turing

class NgramCounts:
    def __init__(self, horizon = 3, load_from = None):
        if load_from is not None:
            self.counts = load_from
        else:
            self.counts = {}
        self.horizon = horizon

    def update(self, seq):
        seq = tuple(seq)
        for i in range(len(seq)):
            for j in range(i + 1, min(i + self.horizon, len(seq))):
                if seq[i:j] not in self.counts:
                    self.counts[seq[i:j]] = 0
                self.counts[seq[i:j]] += 1

class NgramModel:
    def __init__(self, counts, horizon=2, vocab_size=50257):
        self.counts = counts
        self.predictions = {}
        self.vocab_size = vocab_size
        self.horizon = horizon
        self.symbol_list = list(range(vocab_size))

    def pred(self, hist):
        if hist not in self.predictions:
            self.predictions[hist] = torch.zeros(self.vocab_size)

            hist_counts = {
                t: self.counts.counts[hist + (t,)]
                for t in self.symbol_list
                if hist + (t,) in self.counts.counts
            }
            total = sum(hist_counts[t] for t in hist_counts)

            if len(hist_counts) == 0:
                self.predictions[hist] = self.pred(hist[1:])
            else:
                for t in hist_counts:
                    self.predictions[hist][t] = (hist_counts[t] / total) * (1 - 1 / total) #len(hist_counts) / total)

                self.predictions[hist] += self.pred(hist[1:]) * len(hist_counts) / total

            ''' # SGT
            if len(hist_counts) == 0:
                self.predictions[hist] = self.pred(hist[1:])
            else:
                m = max(hist_counts[t] for t in hist_counts)

                if m == 1:
                    # Good-Turing k=1
                    self.predictions[hist] = self.pred(hist[1:])

                elif len(hist_counts) == 1:
                    # Exception for when there is only one example; TODO do this in a more principled way
                    self.predictions[hist][next(hist_counts)] = 1 - 1 / m
                    self.predictions[hist] += self.pred(hist[1:]) / m

                else:
                    sgt = simple_good_turing.SimpleGoodTuring(
                        hist_counts,
                        max(hist_counts[t] for t in hist_counts)
                    )

                    result = sgt.run_sgt(self.symbol_list)
                    for key in result:
                        self.predictions[hist][key] += result[key]

                    # Discounted portion infilled by backfill
                    if len(hist) > 0:
                        self.predictions[hist] += sgt.P0 * self.pred(hist[1:])
            '''

        return self.predictions[hist]

    def generate(self, prompt=(), n=20):
        for _ in range(n):
            dist = self.pred(prompt[-self.horizon:])
            dist /= dist.sum() # To deal with rounding errors
            t = np.random.choice(self.vocab_size, p=dist.numpy())
            prompt += (t,)
        return context.tokenizer.convert_tokens_to_string(
            context.tokenizer.convert_ids_to_tokens(prompt)
        )

    def score(self, seq):
        seq = tuple(seq)
        result = 0
        for i in range(len(seq) - 1):
            dist = self.pred(seq[max(0, i - self.horizon) : i])
            result += torch.log(dist[seq[i + 1]])
        result /= len(seq)
        return result

'''
context = tools.WorkingContext()
def compute_counts():
    tiny = context.load_dataset('tiny')
    counts = NgramCounts(horizon=5)
    for row in tqdm(tiny):
        counts.update(row['sentence'])
    return counts

counts = context.cached_compute(compute_counts, 'ngram-counts-5')

## Test things
model = NgramModel(counts)

for _ in range(20):
    print(model.generate())
'''
