"""
Utility functions for main.ipynb
"""

import numpy as np, re, torch

def preprocess(data: list[dict]):
    """
    Preprocesses and cleans data.

    ## Parameters
    `data`: Data in the form of list of dictionaries

    ## Returns
    `texts`: `np.array` containing external statuses
    `labels`: `np.array` containing internal statuses
    `unique_labels`: `list` containing unique labels
    `vocab`: `list` of unique words in data including padding and unknown tokens
    """
    texts, labels, vocab = [], [], set()
    pattern = re.compile(r"[^\w\s]+")
    for pair in data:

        # Extract external and internal statuses
        external_status, internal_status = pair.values()

        # Strip off white space and lower
        external_status = external_status.strip().lower()
        internal_status = internal_status.strip().lower()

        # Remove any non-alphanumeric characters
        external_status = pattern.sub("", external_status)
        internal_status = pattern.sub("", internal_status)

        # Update vocab
        all_words = external_status.split()
        vocab.update(all_words)

        texts.append(external_status)
        labels.append(internal_status)

    vocab = ["[PAD]", "[UNK]"] + sorted(list(vocab))
    
    return np.array(texts), np.array(labels), np.unique(labels).tolist(), vocab

def train_test_split(inputs, labels, train_ratio, shuffle=False):
    """
    Splits the inputs and labels into train and test sets.

    ## Paramters
    `inputs`: Inputs
    `labels`: Labels
    `train_ratio`: Fraction of train set
    `shuffle`: Shuffle data before splitting

    ## Returns
    `train_inputs`: Inputs in train set
    `train_labels`: Labels in train set
    `test_inputs`: Inputs in test set
    `test_labels`: Labels in test set
    """
    size = len(inputs)
    if size != len(labels):
        raise Exception("Inputs and labels have different lengths")
    if shuffle:
        permutation = np.random.permutation(size)
        inputs = inputs[permutation]
        labels = labels[permutation]
    split_index = int(train_ratio * size)
    return inputs[:split_index], labels[:split_index], inputs[split_index:], labels[split_index:]

def tokenize(texts, vocab, max_tokens):
    """
    Converts a list of texts to a tensor.

    ## Parameters
    `texts`: `list` containing texts
    `vocab`: `list` containing unique words in texts
    `max_tokens`: Maximum tokens in tensor

    ## Returns
    `tensors`: 2D tensor containing vectorized texts
    """
    texts = list(texts)
    tensors = []
    max_len = -1
    for text in texts:
        tensor = []
        for i, word in enumerate(text.split()):
            if i == max_tokens: break
            tensor.append(vocab.index(word) if word in vocab else 1)
        max_len = max(max_len, len(tensor))
        for tensor in tensors:
            tensor += [0] * (max_len - len(tensor))
        tensors.append(tensor)
    
    return torch.tensor(tensors)

def labels2tensor(labels, unique_labels):
    """
    Converts a list of labels to a tensor.

    ## Parameters
    `labels`: `list` containing labels
    `unique_labels`: `list` containing unique_labels

    ## Returns
    `tensor`: Tensor containing vectorized labels
    """
    labels = list(labels)
    tensor = []
    for label in labels:
        tensor.append(unique_labels.index(label))
    return torch.tensor(tensor)

def dynamically_batch(texts, labels, batch_size, vocab, unique_labels, max_tokens):
    """
    Dynamically batches texts and labels.

    ## Parameters
    `texts`: Texts
    `labels`: Labels
    `batch_size`: Maximum number of texts in a batch
    `vocab`: `list` containing unique words in texts
    `unique_labels`: `list` containing unique_labels
    `max_tokens`: Maximum tokens in tensor

    ## Returns
    `batches`: Batched and vectorized texts and labels
    """
    lengths_list = list(map(lambda x: len(x.split()), texts))
    sort_idx = np.argsort(lengths_list)
    texts, labels = texts[sort_idx], labels[sort_idx]
    batches = []
    for i in range(0, len(texts), batch_size):
        batch = (
            tokenize(texts[i : i + batch_size], vocab, max_tokens),
            labels2tensor(labels[i : i + batch_size], unique_labels)
        )
        batches.append(batch)
    return batches
