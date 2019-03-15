import os.path as osp
from collections import Counter
import pickle

import spacy
from textacy import extract
from lineflow.core import CsvDataset


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'
SEP_TOKEN = '<sep>'


NLP = spacy.load('en_core_web_sm')


def tokenize(string):
    return [token.text.lower() for token in NLP(string) if not token.is_space]


def extract_events(doc, lemmatize=True):
    events = []
    for event in extract.subject_verb_object_triples(doc):
        if lemmatize:
            event = [item.lemma_ if item else None for item in event]
        else:
            event = [item.text if item else None for item in event]
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
    return {'story': story, 'text': x}


def build_vocab(tokens, cache='vocab.pkl', max_size=500000):
    if not osp.isfile(cache):
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
        with open(cache, 'wb') as f:
            pickle.dump((t2i, words), f)
    else:
        with open(cache, 'rb') as f:
            t2i, words = pickle.load(f)
    return t2i, words


def getdata():
    path = '/home/takeshita/mnt/DATA/NLP/ROC/ROCStories_winter2.csv'
    data = CsvDataset(
            path,
            header=True
            ).map(preprocess)

    tokens = [
            item
            for story in data
            for events in story['story']
            for event in events
            for item in event
            ]

    t2i, words = build_vocab(tokens)
    return data, t2i, words
