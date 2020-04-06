#!/bin/python
import tensorflow as tf
import pandas as pd
import numpy as np


# dane pacjenta
class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


def retClassLabels(iClassCount):
    toRet = []
    iNumb = 0
    while iNumb <= iClassCount ** 2:
        toRet.append("{0:b}".format(iNumb).zfill(iClassCount))
        iNumb += 1
    return toRet


def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data - mean) / std


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


def getHeader(src):
    header = ""
    with open(src, "r") as file:
        header = file.readline()

    toRet = header.split(',')
    toRet[-1] = toRet[-1][:-1]  # remove \n
    return toRet


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,  # Artificially small to make examples easier to show.
        label_name='flags',
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)
    return dataset


def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label


header_num = ['age', 'TSH', 'T3', 'TT4']
true_or_false = ['f', 't']
header_cat = {
        'sex': ['M', 'F'],
        'on thyroxine': true_or_false,
        'query on thyroxine': true_or_false,
        'on antithyroid medication': true_or_false,
        'sick': true_or_false,
        'pregnant': true_or_false,
        'thyroid surgery': true_or_false,
        'I131 treatment': true_or_false,
        'query hypothyroid': true_or_false,
        'lithium': true_or_false,
        'goitre': true_or_false,
        'tumor': true_or_false,
        'hypopituitary': true_or_false,
        'psych': true_or_false,
        'TSH measured': true_or_false,
        'TT4 measured': true_or_false,
        'T4U measured': true_or_false,
        'FTI measured': true_or_false,
        'TBG measured': true_or_false,
        'referral source': ['WEST', 'STMW', 'SVHC', 'SVI', 'SVHD', 'other']
}


# batch_size <- parametr
#
if __name__ == "__main__":
    LABEL_COLUMN = 'flags'
    LABELS = retClassLabels(1)
    header = getHeader('sick.data')
    train = get_dataset('sick.data', column_names=header)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(29),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])

    model.fit(train, epochs=5)