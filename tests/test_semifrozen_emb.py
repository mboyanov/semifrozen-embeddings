import numpy as np
import torch

from SemiFrozen import SemiFrozenEmbedding


def test_should_return_emb():
    # GIVEN
    num_emb = 100
    emb_dim = 50
    weight = torch.FloatTensor(np.arange(0, 5000).reshape(num_emb, emb_dim))
    emb = SemiFrozenEmbedding(num_emb, emb_dim, padding_idx=1, _weight=weight, frozen_ids=[10,11,12])
    # WHEN
    res = emb(torch.LongTensor([2]))
    # THEN
    np.testing.assert_array_equal(np.arange(100, 150), res.detach().numpy()[0])

def test_should_have_correct_mapping():
    # GIVEN
    num_emb = 100
    emb_dim = 50
    weight = torch.FloatTensor(np.arange(0, 5000).reshape(num_emb, emb_dim))
    # WHEN
    emb = SemiFrozenEmbedding(num_emb, emb_dim, padding_idx=1, _weight=weight, frozen_ids=[10,11,12])
    # THEN
    assert (emb.trainable_map[[10, 11, 12]] == torch.LongTensor([0,0,0])).all()
    assert (emb.frozen_map[[10, 11, 12]] == torch.LongTensor([1,2,3])).all()
    assert (emb.trainable_map[0] == 1)
    assert (emb.frozen_map[0] == 0)


def test_should_have_correct_mapping__pad_zero():
    # GIVEN
    num_emb = 100
    emb_dim = 50
    weight = torch.FloatTensor(np.arange(0, 5000).reshape(num_emb, emb_dim))
    # WHEN
    emb = SemiFrozenEmbedding(num_emb, emb_dim, padding_idx=0, _weight=weight, frozen_ids=[10,11,12])
    # THEN
    assert (emb.trainable_map[[10, 11, 12]] == torch.LongTensor([0,0,0])).all()
    assert (emb.frozen_map[[10, 11, 12]] == torch.LongTensor([1,2,3])).all()
    assert (emb.trainable_map[0] == 0)
    assert (emb.trainable_map[1] == 1)
    assert (emb.frozen_map[0] == 0)

def test_should_return_emb__frozen():
    # GIVEN
    num_emb = 100
    emb_dim = 50
    weight = torch.FloatTensor(np.arange(0, 5000).reshape(num_emb, emb_dim))
    emb = SemiFrozenEmbedding(num_emb, emb_dim, padding_idx=1, _weight=weight, frozen_ids=[10,11,12])
    # WHEN
    res = emb(torch.LongTensor([12]))
    # THEN
    np.testing.assert_array_equal(np.arange(600, 650), res.detach().numpy()[0])


def test_should_return_emb__mixed():
    # GIVEN
    num_emb = 100
    emb_dim = 50
    weight = torch.FloatTensor(np.arange(0, 5000).reshape(num_emb, emb_dim))
    emb = SemiFrozenEmbedding(num_emb, emb_dim, padding_idx=1, _weight=weight, frozen_ids=[10,11,12])
    # WHEN
    res = emb(torch.LongTensor([1, 2, 12]))
    # THEN
    np.testing.assert_array_equal(np.arange(600, 650), res.detach().numpy()[2])
    np.testing.assert_array_equal(np.arange(100, 150), res.detach().numpy()[1])
    np.testing.assert_array_equal(np.zeros((50)), res.detach().numpy()[0])



def test_should_not_have_correct_grads():
    # GIVEN
    num_emb = 100
    emb_dim = 50
    weight = torch.FloatTensor(np.arange(0, 5000).reshape(num_emb, emb_dim))
    emb = SemiFrozenEmbedding(num_emb, emb_dim, padding_idx=1, _weight=weight, frozen_ids=[10,11,12])
    # WHEN
    res = emb(torch.LongTensor([2, 12]))
    loss = res.sum()
    loss.backward()
    # THEN
    assert emb.frozen_emb.weight.grad is None
    assert emb.trainable_emb.weight.grad is not None