import tensorflow as tf


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
