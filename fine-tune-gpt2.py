from tools import *
from gpt2 import *
from tqdm import tqdm, trange
import numpy as np
import copy

large2 = Dataset('large-2')

# Get pretrained models
tokenizer = get_gpt2_tokenizer('en')
tokenizer.pad_token = tokenizer.eos_token
model = get_gpt2_lm('en')

# Tokenize dataset
def batch_tokenize(sentences, batch_size = 8):
    sentences = list(sentences)
    np.random.shuffle(sentences)
    return [
        tokenizer(sentences[i*batch_size:(i + 1)*batch_size], truncation=True, max_length=256, padding=True, return_tensors='pt')
        for i in trange(len(sentences) // batch_size + 1)
    ]

sentence_prop = InjectedProperty('sentence')

batch_prop = Property(
    [sentence_prop],
    'batch-tokens-b8-l256-pt',
    (lambda d: batch_tokenize(d[sentence_prop]))
)

batches = large2[batch_prop]

def finetune(model, batches, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
        for inputs in tqdm(batches):
            optimizer.zero_grad()
            inputs = {k: inputs[k].cuda() for k in inputs}
            outputs = model(**inputs, labels=inputs['input_ids'])
            outputs.loss.backward()
            optimizer.step()
    return model

finetune_prop = Property(
    [batch_prop],
    'finetuned',
    (lambda d: finetune(model, d[batch_prop]))
)

finetuned = large2[finetune_prop]
