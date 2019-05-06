from fastai.text import RNNLearner

from data.tenKgerman import get_german_db
from model import get_model
from model.embeddings import EmbeddingsFactory
from model.embeddings import build_embedding_layer

print("Getting data")
data = get_german_db()
print("Building embeddings")
embeddings = EmbeddingsFactory.get(data.vocab, type='fasttext', lang='de')
missing_ids = []
embeddings_layer = build_embedding_layer(embeddings, missing_ids)

print("Building model")
model = get_model(embeddings_layer, data.c)
print(model)


learn = RNNLearner(data, model)
learn.fit(1)
