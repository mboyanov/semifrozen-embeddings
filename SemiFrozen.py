import torch
import torch.nn as nn


class SemiFrozenEmbedding(nn.Module):

    def __init__(self, num_emb, emb_dim, padding_idx, _weight, frozen_ids = None):
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
        frozen_weight = torch.cat( [torch.zeros(padding_idx+1, emb_dim), _weight[self.frozen_ids]], dim=0)
        self.frozen_emb = nn.Embedding(len(self.frozen_ids)+padding_idx+1, emb_dim, padding_idx=padding_idx, _weight=frozen_weight)
        self.frozen_emb.weight.requires_grad = False
        trainable_ids = [x for x in range(num_emb) if x not in frozen_id_set]
        trainable_weight = torch.cat([torch.zeros(padding_idx + 1, emb_dim), _weight[trainable_ids]], dim=0)
        self.trainable_emb = nn.Embedding(num_emb - len(self.frozen_ids) + padding_idx + 1, emb_dim, padding_idx=padding_idx, _weight=trainable_weight)
        self.trainable_map = torch.zeros(num_emb, dtype=torch.long) + padding_idx
        self.frozen_map = torch.zeros(num_emb, dtype=torch.long) + padding_idx
        frozen_idx = padding_idx + 1
        trainable_idx = padding_idx + 1
        for i in range(num_emb):
            if i == padding_idx:
                trainable_idx += 1
                continue
            elif i in frozen_id_set:
                self.frozen_map[i] = frozen_idx
                frozen_idx += 1
            else:
                self.trainable_map[i] = trainable_idx
                trainable_idx += 1

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