from tools import *
from gpt2 import *
from transformers import GPT2Tokenizer
from tqdm import tqdm, trange
import numpy as np
import torch

sentence_prop = InjectedProperty('sentence')

def tokenized_prop_for(s_prop):
    def tokenize(d):
        result = []
        for s in tqdm(d[s_prop]):
            result.append([
                tokenizer(s)['input_ids']
            ])
        return result

    return Property(
        [s_prop],
        s_prop.name + '_tok',
        tokenize
    )

class LSTM_LM(torch.nn.Module):
    def __init__(self, vocab = 50257, layers = 2, hidden_size = 1024, dropout = 0.2):
        super(LSTM_LM, self).__init__()
        self.vocab = vocab
        self.layers = layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.embed = torch.nn.Embedding(
            num_embeddings = vocab,
            embedding_dim = hidden_size
        )
        self.lstm = torch.nn.LSTM(
            input_size = hidden_size,
            hidden_size = hidden_size,
            num_layers = layers,
            dropout = dropout
        )
        self.out = torch.nn.Linear(
            hidden_size,
            vocab
        )

    def forward(self, inp, state):
        embed = torch.nn.utils.rnn.PackedSequence(
            self.embed(inp.data),
            inp.batch_sizes
        )
        output, state = self.lstm(embed, state)
        result = torch.nn.utils.rnn.PackedSequence(
            self.out(output.data),
            output.batch_sizes
        )
        return result, state

    def init_state(self, batch):
        return (
            torch.zeros(self.layers, batch, self.hidden_size),
            torch.zeros(self.layers, batch, self.hidden_size)
        )

def packed_seq(batch):
    if any(len(s) <= 1 for s in batch):
        return None, None

    return (
        # Inputs
        torch.nn.utils.rnn.pack_sequence([
            torch.LongTensor(seq[:-1]) for seq in batch
        ]),
        # Targets
        torch.nn.utils.rnn.pack_sequence([
            torch.LongTensor(seq[1:]) for seq in batch
        ])
    )

def lstm_prop(s_prop,
        seed = 0,
        hidden_size = 1024,
        layers = 2,
        dropout = 0.2,
        batch_size = 32,
        epochs = 5
    ):

    t_prop = tokenized_prop_for(s_prop)

    def train_lstm(d):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = LSTM_LM(
            layers = layers,
            hidden_size = hidden_size,
            dropout = dropout
        ).cuda()

        sorted_sentences = sorted(
            [x[0] for x in d[t_prop] if 1 < len(x[0]) < 512], # Limit batch size
            key = lambda x: -len(x)
        )
        batches = [
            packed_seq(sorted_sentences[i:i + batch_size])
            for i in trange(len(sorted_sentences) // batch_size + 1)
        ]

        criterion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr = 3e-4)

        for _ in range(epochs):
            np.random.shuffle(batches)
            running_loss = 0
            i = 0

            for batch, target in tqdm(batches):
                if batch is None:
                    continue
                batch, target = batch.cuda(), target.cuda()
                output, state = model(
                    batch,
                    tuple(
                        x.cuda() for x in model.init_state(batch.batch_sizes[0])
                    )
                )
                loss = criterion(
                    output.data,
                    target.data
                )
                running_loss = running_loss * 0.95 + 0.05 * (loss.detach().cpu())
                optim.zero_grad()
                loss.backward()
                optim.step()
                if i % 1000 == 0:
                    print('Running loss: %f' % running_loss)
                i += 1
            '''
            for sliceable_batch, lengths in tqdm(batches):
                state = tuple(
                    x.cuda()
                    for x in model.init_state(sliceable_batch.shape[0])
                )
                seq_len = sliceable_batch.shape[1]

                optim.zero_grad()
                loss = 0
                for i in range(seq_len - 1):
                    output, state = model(
                        sliceable_batch[:, i].cuda(),
                        state
                    )

                    loss += criterion(
                        output[:lengths[i + 1]],
                        sliceable_batch[:, i + 1][:lengths[i + 1]]
                    )
                loss.backward()
                optim.step()
            '''

            return model

    return Property(
        [t_prop],
        s_prop.name + '_lstm_%d_%d_%d_%f_%d_%d' % (seed, hidden_size, layers, dropout, batch_size, epochs),
        train_lstm
    )
