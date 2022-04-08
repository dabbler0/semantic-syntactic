from tools import *
import nltk
from nltk import tokenize
import datasets

nltk.download('averaged_perceptron_tagger')

# Simple caching infrastructure
root_dir = '/raid/lingo/abau/slim-cache'
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

# Create datasets of various sizes
news_dataset = datasets.load_dataset('cc_news')['train']

for i in range(3):
    # Divide into sentences
    tiny = news_dataset.shard(num_shards=500, index=i)
    tiny = tiny.map(lambda x: { 'sentences': tokenize.sent_tokenize(x['text']) })

    # Flatten into sentences
    def process_batch(batch, indices):
        results = { 'sentence': [], 'document': [], 'offset': [] }

        for item, document in zip(batch['sentences'], indices):
            for offset, sentence in enumerate(item):
                results['sentence'].append(sentence)
                results['document'].append(document)
                results['offset'].append(offset)
        return results

    flat = tiny.map(
        process_batch,
        batched=True,
        with_indices=True,
        batch_size=64,
        remove_columns=['title', 'text', 'domain', 'date', 'description', 'url', 'image_url', 'sentences']
    )

    create_dataset('tiny-%d' % i, {
        'sentence': flat['sentence'],
        'document': flat['document'],
        'offset': flat['offset']
    })

for i in range(3):
    # Divide into sentences
    tiny = news_dataset.shard(num_shards=10, index=i)
    tiny = tiny.map(lambda x: { 'sentences': tokenize.sent_tokenize(x['text']) })

    # Flatten into sentences
    def process_batch(batch, indices):
        results = { 'sentence': [], 'document': [], 'offset': [] }

        for item, document in zip(batch['sentences'], indices):
            for offset, sentence in enumerate(item):
                results['sentence'].append(sentence)
                results['document'].append(document)
                results['offset'].append(offset)
        return results

    flat = tiny.map(
        process_batch,
        batched=True,
        with_indices=True,
        batch_size=64,
        remove_columns=['title', 'text', 'domain', 'date', 'description', 'url', 'image_url', 'sentences']
    )

    create_dataset('large-%d' % i, {
        'sentence': flat['sentence'],
        'document': flat['document'],
        'offset': flat['offset']
    })
