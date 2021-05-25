# load libraries
import matplotlib.pyplot as plt
import os 
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

print(tf.__version__)

# load dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')


dataset_dir = 'G:/rauf/STEPBYSTEP/Tutorial/TensorFlow2/Simple_Text_Classification(Sentiment Analysis)/Simple_Text_Classification(Sentiment Analysis)/aclImdb'

os.listdir(dataset_dir)

train_dir = 'G:/rauf/STEPBYSTEP/Tutorial/TensorFlow2/Simple_Text_Classification(Sentiment Analysis)/Simple_Text_Classification(Sentiment Analysis)/aclImdb/train'
test_dir = 'G:/rauf/STEPBYSTEP/Tutorial/TensorFlow2/Simple_Text_Classification(Sentiment Analysis)/Simple_Text_Classification(Sentiment Analysis)/aclImdb/test'

sample_example_dir = 'G:/rauf/STEPBYSTEP/Tutorial/TensorFlow2/Simple_Text_Classification(Sentiment Analysis)/Simple_Text_Classification(Sentiment Analysis)/aclImdb/train/pos/0_9.txt'

# to check dataset
with open(sample_example_dir) as f:
    print(f.read())




# remove unnecesary file
remove_dir = 'G:/rauf/STEPBYSTEP/Tutorial/TensorFlow2/Simple_Text_Classification(Sentiment Analysis)/Simple_Text_Classification(Sentiment Analysis)/aclImdb/train/unsup'
shutil.rmtree(remove_dir)


# separate and load dataset for training
batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size)

# preprocessing dataset
def customstandardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

# standarize dataset
max_features = 10000
sequence_length = 250

vectorize_layer = TextVectorization(
    standardize=customstandardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

text_batch, label_batch = next(iter(raw_train_ds))
first_rewiev, first_label = text_batch[0], label_batch[0]
print("Review: ", first_rewiev)
print("Label: ", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_rewiev, first_label))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


# configure dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# crete the model
embedding_dim = 16
model =tf.keras.Sequential([
    layers.Embedding(max_features+1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
    ])

model.summary()

# compile the model
model.compile(loss=losses.binary_crossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# train the model
epochs = 10
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# evaluate the model
loss, accuracy = model.evaluate(test_ds)
print("Losess: ", loss)
print("Accuracy: ", accuracy)

# deploy the model
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
    ])

export_model.compile(
    loss=losses.binary_crossentropy(from_logits=False), optimizer='adam', metrics=['accuracy']
    )

loss, accuracy = export_model.evaluate(raw_test_dss)
print(accuracy)

examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

export_model.predict(examples)
