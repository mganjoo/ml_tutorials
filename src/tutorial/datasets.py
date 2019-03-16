import numpy as np
import os
import pandas as pd
import random
from tensorflow.keras import utils


def load_iris_dataset():
    """Loads the Iris dataset."""
    url = ("https://archive.ics.uci.edu/ml/" +
           "machine-learning-databases/iris/iris.data")
    numeric_features = ['sepal_length', 'sepal_width',
                        'petal_length', 'petal_width']
    csv_column_names = numeric_features + ['species']
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    data_path = utils.get_file("iris_data.csv", url)

    return pd.read_csv(data_path, names=csv_column_names), species


def load_imdb_dataset(base_path, seed=123):
    """Loads the IMDb movie reviews sentiment analysis dataset.

    # Arguments
        base_path: string, path to the base data directory (base).
        seed: int, seed for randomizer.

    # Returns
        A tuple of data (with keys 'train' and 'test') and labels (same keys).
        Number of training samples: 25000
        Number of test samples: 25000
        Number of categories: 2 (0 - negative, 1 - positive)

    # References
        Mass et al., http://www.aclweb.org/anthology/P11-1015
    """
    imdb_data_path = os.path.join(base_path, 'aclImdb')

    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)

    # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ({'train': train_texts, 'test': test_texts},
            {'train': np.array(train_labels), 'test': np.array(test_labels)})
