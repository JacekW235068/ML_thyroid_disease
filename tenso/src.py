import functools
from _lsprof import profiler_entry

import numpy as np
import tensorflow as tf

import pandas as pd

# labal value to predict
LABEL_COLUMN = 'flags'


def getHeader(src):
    header = ""
    with open(src, "r") as file:
        header = file.readline()

    toRet = header.split(',')
    toRet[-1] = toRet[-1][:-1]  # remove \n
    return


def show_batch(dataset):
    print("BATCH START ------- \n")
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))
    print("\nBATCH -- END")


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=10,  # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)
    return dataset


def pack(features, label):
    print(features.values(), " ", label, '\n')
    return tf.stack(list(features.values()), axis=-1), label


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features
        return features, labels


def normalize_numeric_data(data, mean, std):
    # Center the data
    std[std == 0] = 1
    return (data - mean) / std


np.set_printoptions(precision=3, suppress=True)

# data from file using copy paste function from tutorial
raw_train_data = get_dataset('sick.data')
raw_test_data = get_dataset('test.data')

show_batch(raw_train_data)
print("-----------------------------")

# DATA PREPROCESSING

SELECT_COLUMNS = ['flags', 'age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
DEFAULTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
temp_dataset = get_dataset('sick.data',
                           select_columns=SELECT_COLUMNS,
                           column_defaults=DEFAULTS)

show_batch(temp_dataset)
print(" BATCH END -----------------------------")

packed_dataset = temp_dataset.map(pack)

for features, labels in packed_dataset.take(1):
    print(features.numpy())
    print()
    print(labels.numpy())
print("-----------------------------")

example_batch, labels_batch = next(iter(temp_dataset))

NUMERIC_FEATURES = ['age', 'TSH', 'T3', 'TT4', 'FTI', 'TBG', 'T4U']

packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

show_batch(packed_train_data)
print("-----------------------------")

example_batch, labels_batch = next(iter(packed_train_data))
desc = pd.read_csv('sick.data')[NUMERIC_FEATURES].describe()

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

# See what you just created.
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]

print(example_batch['numeric'])
print("-----------------------------")
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
print(numeric_layer(example_batch).numpy())
print("-----------------------------")
true_or_false = ['f', 't']
CATEGORIES = {
    'sex': ['M', 'F', '0'],
    'on-thyroxine': true_or_false,
    'query-on-thyroxine': true_or_false,
    'on-antithyroid-medication': true_or_false,
    'sick': true_or_false,
    'pregnant': true_or_false,
    'thyroid-surgery': true_or_false,
    'I131-treatment': true_or_false,
    'query-hypothyroid': true_or_false,
    'lithium': true_or_false,
    'goitre': true_or_false,
    'tumor': true_or_false,
    'hypopituitary': true_or_false,
    'psych': true_or_false,
    'TSH-measured': true_or_false,
    'TT4-measured': true_or_false,
    'T4U-measured': true_or_false,
    'FTI-measured': true_or_false,
    'TBG-measured': true_or_false,
    'referral-source': ['WEST', 'STMW', 'SVHC', 'SVI', 'SVHD', 'other']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

print(categorical_columns)
print("-----------------------------")

categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print(categorical_layer(example_batch).numpy()[0])
print("-----------------------------")

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)

print(preprocessing_layer(example_batch).numpy()[0])
print("-----------------------------")

# MODEL

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='SGD',
    metrics=['accuracy'])

train_data = packed_train_data
test_data = packed_test_data
generated = list(test_data)
model.fit(train_data, epochs=40)

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
show_batch(test_data)

predictions = model.predict(test_data, batch_size=1, steps=1)

for predin, labels in zip(predictions, generated[0][1]):
    lv_sigmoid = tf.sigmoid(predin)
    print(lv_sigmoid, 'for', labels)
