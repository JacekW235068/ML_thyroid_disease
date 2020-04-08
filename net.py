import functools
import pandas as pd
import numpy as np
import tensorflow as tf

import pandas as pd
np.set_printoptions(precision=3, suppress=True)
def show_batch(dataset):
  for batch, _ in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))
def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
  return dataset

def pack(features, label):
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

LABEL_COLUMN = 'flags'
file_path = "tenso/sick.data"

raw_train_data = get_dataset(file_path)
show_batch(raw_train_data)
print ("-----------------------------------------")
SELECT_COLUMNS = ['age','TSH', 'T3', 'TT4', 'T4U', 'FTI','flags']
DEFAULTS = [0.0,0.0, 0.0, 0.0, 0.0, 0.0,0]
temp_dataset = get_dataset(file_path, 
                           select_columns=SELECT_COLUMNS,
                           column_defaults = DEFAULTS)

show_batch(temp_dataset)
print ("----------Numeric columns examples-------------------------------")
packed_dataset = temp_dataset.map(pack)

for features, labels in packed_dataset.take(1):
  print(features.numpy())
  print()
  print(labels.numpy())
print ("-----------------------------------------")

NUMERIC_FEATURES = ['age','TSH', 'T3', 'TT4', 'T4U', 'FTI']

packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

show_batch(packed_train_data)
print ("-----------------------------------------")

desc = pd.read_csv(file_path)[NUMERIC_FEATURES].describe()

print (desc)

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])