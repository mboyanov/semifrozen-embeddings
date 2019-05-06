import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText
from fastai.text import Vocab
import torch
import torch.nn as nn


def from_glove_and_vocab(path, vocab: Vocab):
    glove_loader = lambda: KeyedVectors.load_word2vec_format(path)
    unk_handler = MeanStdUnKHandler()
    return load_wv(glove_loader, vocab, unk_handler=unk_handler)


class MeanStdUnKHandler:
    def __init__(self):
        self.mean = None
        self.std = None

    def __call__(self, wv, w):
        if self.mean is None:
            self.mean = wv.vectors.mean(axis=0)
            self.std = wv.vectors.std()
        return (self.mean + np.random.randn(self.mean.shape[0]) * self.std)[None]


class FasttextUnkHandler:

    def __init__(self):
        pass

    def __call__(self, wv: FastText, w):
        if w in wv:
            return wv[w][None]
        else:
            return np.random.randn(wv.vector_size)[None]


def load_wv(loader, vocab, unk_handler):
    wv = loader()
    missing = []
    pretrained = []
    for i, w in enumerate(vocab.itos):
        if w in wv.wv.vocab.keys():
            pretrained.append(wv[w][None])
        else:
            pretrained.append(unk_handler(wv, w))
            missing.append(i)
    pretrained = np.vstack(pretrained)
    return pretrained, missing


def from_fasttext_and_vocab(path, vocab: Vocab):
    fasttext_loader =  lambda : FastText.load_fasttext_format(path)
    unk_handler = FasttextUnkHandler()
    return load_wv(fasttext_loader, vocab, unk_handler)

class EmbeddingsFactory:
    pathmap = {
        'fasttext_de': "/data/fasttext/cc.de.300.bin",
    }

    @staticmethod
    def get(vocab: Vocab, type='fasttext', lang='de', return_missing=False):
        pretrained, missing_ids = from_fasttext_and_vocab(path=EmbeddingsFactory.pathmap[f"{type}_{lang}"], vocab=vocab)
        pretrained = torch.FloatTensor(pretrained)
        if return_missing:
            return pretrained, missing_ids
        return pretrained


class SemiFrozenEmbedding(nn.Module):

    def __init__(self, num_emb, emb_dim, padding_idx, _weight, frozen_ids=None):
        """
        Acts as a replacement for the nn.Embedding module where some of the vectors for the frozen_ids are kept frozen.
        :param num_emb:
        :param emb_dim:
        :param padding_idx:
        :param _weight:
        :param frozen_ids:
        """
        frozen_ids = frozen_ids if frozen_ids is not None else []
        frozen_id_set = set(frozen_ids)
        self.embedding_dim, self.num_emb = emb_dim, num_emb
        super(SemiFrozenEmbedding, self).__init__()
        self.frozen_ids = frozen_ids
        internal_padding_idx = 0
        trainable_ids = [x for x in range(num_emb) if x not in frozen_id_set]
        if padding_idx in trainable_ids: trainable_ids.remove(padding_idx)
        if padding_idx in frozen_ids: frozen_ids.remove(padding_idx)

        frozen_weight = torch.cat([torch.zeros(1, emb_dim), _weight[self.frozen_ids]], dim=0)
        self.frozen_emb = nn.Embedding(len(self.frozen_ids) + 1, emb_dim, padding_idx=internal_padding_idx,
                                       _weight=frozen_weight)
        self.frozen_emb.weight.requires_grad = False

        trainable_weight = torch.cat([torch.zeros(1, emb_dim), _weight[trainable_ids]], dim=0)

        self.trainable_emb = nn.Embedding(len(trainable_ids) + 1, emb_dim, padding_idx=internal_padding_idx,
                                          _weight=trainable_weight)

        self.trainable_map = torch.zeros(num_emb, dtype=torch.long) + internal_padding_idx
        self.frozen_map = torch.zeros(num_emb, dtype=torch.long) + internal_padding_idx

        for idx, frozen_idx in enumerate(frozen_ids):
            self.frozen_map[frozen_idx] = idx + 1
        for idx, trainable_idx in enumerate(trainable_ids):
            self.trainable_map[trainable_idx] = idx + 1

    def forward(self, *input):
        text_input = input[0]
        self.trainable_map = self.trainable_map.to(text_input.device)
        self.frozen_map = self.frozen_map.to(text_input.device)
        trainable = self.trainable_emb(self.trainable_map[text_input])
        frozen = self.frozen_emb(self.frozen_map[text_input])
        return trainable + frozen

    def semi_freeze(self):
        self.frozen_emb.weight.requires_grad = False

    def semi_unfreeze(self):
        self.frozen_emb.weight.requires_grad = True


def build_embedding_layer(embeddings: np.array, missing_ids: list) -> SemiFrozenEmbedding:
    frozen_ids = [x for x in range(embeddings.shape[0]) if x not in missing_ids]
    return SemiFrozenEmbedding(embeddings.shape[0], embeddings.shape[1], padding_idx=1, _weight=embeddings,
                               frozen_ids=frozen_ids)
