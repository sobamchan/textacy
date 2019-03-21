from pathlib import Path
import pickle
from collections import Counter

import fire
import spacy
from textacy import extract
from lineflow.core import CsvDataset


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'
SEP_TOKEN = '<sep>'
NONE_TOKEN = '<none>'


NLP = spacy.load('en_core_web_sm')


def tokenize(string):
    return [token.text.lower() for token in NLP(string) if not token.is_space]


def extract_events(doc, lemmatize=True):
    events = []
    for event in extract.subject_verb_object_triples(doc):
        if lemmatize:
            event = [item.lemma_ if item else NONE_TOKEN for item in event]
        else:
            event = [item.text if item else NONE_TOKEN for item in event]
        events.append(event)
    return events


def preprocess(x):
    contexts = (
            x['sentence1'].lower(),
            x['sentence2'].lower(),
            x['sentence3'].lower(),
            x['sentence4'].lower(),
            x['sentence5'].lower()
            )
    story = [extract_events(NLP(context)) for context in contexts]

    events = []
    # Merge events
    for line in story:
        event = []
        event += [START_TOKEN]
        for tokens in line:
            event += tokens
            event += [SEP_TOKEN]
        event = event[:-1]
        event += [END_TOKEN]
        events.append(event)

    return events


def postprocess(t2i):
    def _f(events):
        return [[t2i.get(e, UNK_TOKEN) for e in event] for event in events]
    return _f


def build_vocab(tokens, max_size=500000):
    counter = Counter(tokens)
    words, _ = zip(*counter.most_common(max_size))
    words = [PAD_TOKEN, UNK_TOKEN, SEP_TOKEN] + list(words)
    t2i = dict(zip(words, range(len(words))))
    if START_TOKEN not in t2i:
        t2i[START_TOKEN] = len(t2i)
        words += [START_TOKEN]
    if END_TOKEN not in t2i:
        t2i[END_TOKEN] = len(t2i)
        words += [END_TOKEN]
    if PAD_TOKEN not in t2i:
        t2i[PAD_TOKEN] = len(t2i)
        words += [PAD_TOKEN]
    if SEP_TOKEN not in t2i:
        t2i[SEP_TOKEN] = len(t2i)
        words += [SEP_TOKEN]
    if NONE_TOKEN not in t2i:
        t2i[NONE_TOKEN] = len(t2i)
        words += [NONE_TOKEN]
    return t2i, words


def build(dpath, savedir):
    savedir = Path(savedir)

    # path = '/home/takeshita/mnt/DATA/NLP/ROC/ROCStories_winter2.csv'
    dataset = CsvDataset(
            dpath,
            header=True
            ).map(preprocess)

    tokens = [
            item
            for story in dataset
            for event in story
            for item in event
            ]

    t2i, words = build_vocab(tokens)
    dataset = dataset.map(postprocess(t2i))
    dataset.save(savedir / 'dataset.token.pkl')
    with open(savedir / 'vocab.pkl', 'wb') as f:
        pickle.dump((t2i, words), f)


if __name__ == '__main__':
    fire.Fire()
