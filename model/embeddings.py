import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText


from fastai.text import Vocab
import fasttext

def from_glove_and_vocab(path, vocab:Vocab):
    wv = KeyedVectors.load_word2vec_format(path)
    mean = wv.vectors.mean(axis=0)
    std = wv.vectors.std()
    missing = []
    missing_words = []
    pretrained = []
    for i, w in enumerate(vocab.itos):
        if w in wv:
            pretrained.append(wv[w][None])
        else:
            pretrained.append((mean + np.random.randn(mean.shape[0]) * std)[None])
            missing.append(i)
            missing_words.append(w)
    pretrained = np.vstack(pretrained)
    return pretrained


def from_fasttext_and_vocab(path, vocab:Vocab):
    wv = FastText.load_fasttext_format(path)
    missing = []
    missing_words = []
    pretrained = []
    for i, w in enumerate(vocab.itos):
        if w in wv.wv.vocab.keys():
            pretrained.append(wv[w][None])
        else:
            pretrained.append(wv[w][None])
            missing.append(i)
            missing_words.append(w)
    pretrained = np.vstack(pretrained)
    return pretrained


#print(from_glove_and_vocab('/data/embeddings/glove.100k.300d.txt', Vocab(['hello', 'world','gkasogkas[oga'])))
#print(from_fasttext_and_vocab('/data/fasttext/cc.de.300.bin', Vocab(['hello', 'world','gkasogkas[oga'])))