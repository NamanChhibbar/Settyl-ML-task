"""
Utility functions for main.ipynb
"""

import numpy as np, re

def preprocess(data: list[dict]):
    """
    Preprocesses and cleans data.

    ## Parameters
    `data`: Data in the form of list of dictionaries

    ## Returns
    `inputs`: `np.array` containing external statuses
    `labels`: `np.array` containing internal statuses
    `unique_labels`: `list` containing unique labels
    `vocab`: `list` of unique words in data
    """
    inputs, labels, vocab = [], [], set()
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

        inputs.append(external_status)
        labels.append(internal_status)
    
    return np.array(inputs), np.array(labels), np.unique(labels).tolist(), sorted(list(vocab))
