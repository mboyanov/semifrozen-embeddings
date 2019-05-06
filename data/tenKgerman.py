from fastai.text import Tokenizer, TokenizeProcessor, NumericalizeProcessor, TextList

def get_german_db(bs=32):
    tokenizer = Tokenizer(lang='de')
    tokenizer_proc = TokenizeProcessor(tokenizer=tokenizer)
    num_proc = NumericalizeProcessor()
    processor = [tokenizer_proc, num_proc]
    db = TextList\
        .from_csv('/data/10kgerman', 'train.csv', cols=["text"], processor=processor)\
        .split_by_rand_pct(0.1)\
        .label_from_df('c').databunch()
    db.batch_size = bs
    return db