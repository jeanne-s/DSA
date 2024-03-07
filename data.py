from datasets import load_dataset

def get_data_sample(split: str = 'train', 
                    n_samples: int = 1,
                    dataset_name: str = 'wikitext',
                    seed: int = 42):

    if dataset_name.__contains__('wiki'):
        dataset = load_dataset("iohadrubin/wikitext-103-raw-v1", split=split, streaming=True)
    elif dataset_name.__contains__('tiny'):
        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}.")

    shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=1_000)
    dataset_iter = iter(shuffled_dataset)
    samples = []
    
    for _ in range(n_samples):
        try:
            samples.append(next(dataset_iter)['text'])
        except StopIteration:
            break

    return samples