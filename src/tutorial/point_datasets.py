import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

_NUM_SAMPLES = 500


def gaussian(num_points=_NUM_SAMPLES):
    c0_center = (2, 2)
    c1_center = (-2, -2)
    variance = np.sqrt(0.5)
    x0 = np.concatenate((
        np.random.normal(c0_center[0], variance, num_points//2),
        np.random.normal(c1_center[0], variance, num_points//2)))
    x1 = np.concatenate((
        np.random.normal(c0_center[1], variance, num_points//2),
        np.random.normal(c1_center[1], variance, num_points//2)))
    y = np.concatenate((np.zeros(num_points//2), np.ones(num_points//2)))
    return pd.DataFrame({'x0': x0, 'x1': x1, 'y': y})


def xor(num_points=_NUM_SAMPLES):
    pad = np.vectorize(lambda x: x + 0.3 if x > 0 else x - 0.3)
    x0 = pad(np.random.random(num_points) * 10 - 5)
    x1 = pad(np.random.random(num_points) * 10 - 5)
    y = (x0 * x1 < 0).astype(int)
    return pd.DataFrame({'x0': x0, 'x1': x1, 'y': y})


def spiral(num_points=_NUM_SAMPLES):
    def gen_spiral(n, delta_t, noise=0.05):
        r = np.linspace(0, 5, num=n, endpoint=False)
        t = np.linspace(0, 1.75, num=n, endpoint=False) * 2*np.pi + delta_t
        x0 = r * np.sin(t) + (np.random.random(n)*2 - 1) * noise
        x1 = r * np.cos(t) + (np.random.random(n)*2 - 1) * noise
        return (x0, x1)

    (x0_0, x1_0) = gen_spiral(num_points//2, np.pi)
    (x0_1, x1_1) = gen_spiral(num_points//2, 0)
    x0 = np.concatenate((x0_0, x0_1))
    x1 = np.concatenate((x1_0, x1_1))
    y = np.concatenate((np.zeros(num_points//2), np.ones(num_points//2)))
    return pd.DataFrame({'x0': x0, 'x1': x1, 'y': y})


def plot_dataset(dataset):
    sns.lmplot(x='x0', y='x1', data=dataset, hue='y',
               fit_reg=False, legend=False)


def train_input_fn(features, labels, batch_size):
    """An input function for training."""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    assert batch_size is not None, "batch_size must not be None"

    # Shuffle, repeat, and batch the examples.
    # This ensures that the examples are read in a random order, and
    # also that the data set loops through once the training function
    # reads through all examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction."""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples.
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
