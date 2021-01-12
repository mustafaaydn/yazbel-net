# Kullanıcı adından hareketle, kullanıcının yanıtlarının halihazırda diske
# kaydedildiğini varsayıp ilgili dosyayı bir veri setine çevirmeye yarayan
# fonksiyonlar dizisinin olduğu yer.

import os

import numpy as np
import tensorflow as tf


def _get_text(username):
    """
    Given the username, open up the corresponding file in ./replies dir and
    slurp.

    Parameters
    -----------
    username: str
        the username whose replies are wanted

    Returns
    --------
        The contents of the file read
    """
    path_to_text = os.path.join("replies", f"{username}.txt")
    with open(path_to_text, "r", encoding="utf-8") as fh:
        text = fh.read()
    return text


def _input_target_maker(seq):
    """
    Kind of a sliding window on a string with sequence stride being 1 and
    sampling rate also being 1. e.g. "tensorflow" comes "tensorflo" and
    "ensorflow" return.

    Parameters
    ------------
    seq: str
        The sequence to split into `input` and `target`

    Returns
    ---------
        2-tuple of (input part, target part)
    """
    inp_txt = seq[:-1]
    tar_txt = seq[1:]
    return inp_txt, tar_txt


def _prepare_mappers(text):
    """
    Given a text, extracts the vocabulary i.e. unique set of characters and
    makes 2 mappings: from characters to numbers and vice versa. Numbers simply
    start from 0 and end at vocab_size-1.

    Parameters
    -----------
    text: str
        The text under investigation

    Returns
    --------
        2-tuple of mappings (char2num, num2char)
    """
    # preserve order! (>= py3.7)
    vocab = dict.fromkeys(text).keys()

    # Many models don't see words, they see numbers
    char2num = {char: num for num, char in enumerate(vocab)}

    # reverse also needed so we can read the outputs
    num2char = {num: char for char, num in char2num.items()}

    return char2num, num2char


def _train_val_split(dataset, val_frac):
    """
    Splits tf.data.Dataset to training and validation sets
    Adapted and modified from: https://stackoverflow.com/a/58452268/9332187

    Parameters
    ------------
    dataset: tf.data.Dataset
        the whole dataset

    val_frac: float
        the fraction of the validation data

    Returns
    --------
        2-tuple of datasets (training dataset, validation dataset)
    """
    def is_val(x, y):
        return x % 10 < round(10 * val_frac)

    def is_train(x, y):
        return not is_val(x, y)

    def recover(x, y):
        return y

    val_ds = dataset.enumerate() \
                    .filter(is_val) \
                    .map(recover)

    train_ds = dataset.enumerate() \
                      .filter(is_train) \
                      .map(recover)
    return train_ds, val_ds


def make_dataset(username, val_frac=0.1, seq_length=100, batch_size=64):
    """
    Prepares the dataset to train on for the username.

    Parameters
    -----------
    username: str
        should be such that "./replies/{username}.txt" exists

    val_frac: float, optional, default=0.1
        the fraction of the validation data e.g. 0.1 means 10% of data will be
        used for validation and the rest (and only the rest!) for training.
        This being nonzero is crucial to prevent overfitting since early
        stopping is applied whenever validation data exists.

    seq_length: int, optional, default=100
        how many characters should the model look-back whilst training (not
        to be confused by text generation sequence length!). Increasing this
        implies the text has relatively longer temporal dependency and makes
        the model more comprehensive if you will (computation time increases
        of course).

    batch_size: int, optional, default=64
        how many samples should be propagated together in one iteration
    """
    # read in the text and get the relate mappers
    text = _get_text(username)
    char2num, num2char = _prepare_mappers(text)

    # Map the whole text with char2num
    all_text_numed = np.array([char2num[char] for char in text])

    # tensorflow dataset all of a sudden :)
    dataset = tf.data.Dataset.from_tensor_slices(all_text_numed)

    # Input - Target relation is "tensorflo" - "ensorflow", hence the +1
    # The dataset was a long series of integers; now we "batch" sequences
    sequences = dataset.batch(seq_length+1, drop_remainder=True)

    # apply the sliding window-like scheme
    dataset = sequences.map(_input_target_maker)

    # Now we shuffle the data and pack into batches
    # note that this batching is different than batching done previously :)
    dataset = dataset.shuffle(buffer_size=10_000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Split into training and validation based on validation fraction
    train_ds, val_ds = _train_val_split(dataset, val_frac)

    return train_ds, val_ds, (char2num, num2char)
