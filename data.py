from datasets import load_dataset

def get_wikitext_103_sample(split='train', n_samples=1):
    dataset = load_dataset("iohadrubin/wikitext-103-raw-v1", split='train', streaming=True)
    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=1_000)
    return next(iter(shuffled_dataset))['text']