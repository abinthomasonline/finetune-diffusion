from .default import DefaultDataset


def load_dataset(data_dir, tokenizer):
    return DefaultDataset(data_dir, tokenizer)
