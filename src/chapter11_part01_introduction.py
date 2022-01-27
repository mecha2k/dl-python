# This is a companion notebook for the book [Deep Learning with Python, Second Edition]
# (https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff).
# For readability, it only contains runnable code blocks and section titles, and omits everything else in the book:
# text paragraphs, figures, and pseudocode.
#
# This notebook was generated for TensorFlow 2.6.

# Deep learning for text
## Natural-language processing: The bird's eye view
## Preparing text data
### Text standardization
### Text splitting (tokenization)
### Vocabulary indexing
### Using the TextVectorization layer

import string


class Vectorizer:
    def __init__(self):
        self.inverse_vocabulary = None
        self.vocabulary = {"": 0, "[UNK]": 1}

    @staticmethod
    def standardize(text):
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)

    def tokenize(self, text):
        text = self.standardize(text)
        return text.split()

    def make_vocabulary(self, dataset):
        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        self.inverse_vocabulary = dict((v, k) for k, v in self.vocabulary.items())

    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence):
        return " ".join(self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence)


vectorizer = Vectorizer()
dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]
vectorizer.make_vocabulary(dataset)


test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = vectorizer.encode(test_sentence)
print(encoded_sentence)


decoded_sentence = vectorizer.decode(encoded_sentence)
print(decoded_sentence)


from tensorflow.keras.layers import TextVectorization

text_vectorization = TextVectorization(
    output_mode="int",
)


import re
import string
import tensorflow as tf


def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor)
    return tf.strings.regex_replace(lowercase_string, f"[{re.escape(string.punctuation)}]", "")


def custom_split_fn(string_tensor):
    return tf.strings.split(string_tensor)


text_vectorization = TextVectorization(
    output_mode="int",
    standardize=custom_standardization_fn,
    split=custom_split_fn,
)


dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]
text_vectorization.adapt(dataset)


# **Displaying the vocabulary**
text_vectorization.get_vocabulary()


vocabulary = text_vectorization.get_vocabulary()
test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = text_vectorization(test_sentence)
print(encoded_sentence)


inverse_vocab = dict(enumerate(vocabulary))
decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
print(decoded_sentence)


# ## Two approaches for representing groups of words: Sets and sequences
# ### Preparing the IMDB movie reviews data
# get_ipython().system("curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
# get_ipython().system("tar -xf aclImdb_v1.tar.gz")
# get_ipython().system("rm -r aclImdb/train/unsup")
# get_ipython().system("cat aclImdb/train/pos/4077_10.txt")


import os, pathlib, shutil, random

base_dir = pathlib.Path("data/aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"
for category in ("neg", "pos"):
    os.makedirs(val_dir / category)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)
    num_val_samples = int(0.2 * len(files))
    val_files = files[-num_val_samples:]
    for fname in val_files:
        shutil.move(train_dir / category / fname, val_dir / category / fname)


from tensorflow import keras

batch_size = 32

train_ds = keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory("aclImdb/val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=batch_size)


# **Displaying the shapes and dtypes of the first batch**
for inputs, targets in train_ds:
    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    print("inputs[0]:", inputs[0])
    print("targets[0]:", targets[0])
    break


# ### Processing words as a set: The bag-of-words approach
# #### Single words (unigrams) with binary encoding
# **Preprocessing our datasets with a `TextVectorization` layer**
text_vectorization = TextVectorization(
    max_tokens=20000,
    output_mode="multi_hot",
)
text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)

binary_1gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
binary_1gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
binary_1gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)


# **Inspecting the output of our binary unigram dataset**
for inputs, targets in binary_1gram_train_ds:
    print("inputs.shape:", inputs.shape)
    print("inputs.dtype:", inputs.dtype)
    print("targets.shape:", targets.shape)
    print("targets.dtype:", targets.dtype)
    print("inputs[0]:", inputs[0])
    print("targets[0]:", targets[0])
    break


# **Our model-building utility**
from tensorflow import keras
from tensorflow.keras import layers


def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# **Training and testing the binary unigram model**
model = get_model()
model.summary()
callbacks = [keras.callbacks.ModelCheckpoint("binary_1gram.keras", save_best_only=True)]
model.fit(
    binary_1gram_train_ds.cache(),
    validation_data=binary_1gram_val_ds.cache(),
    epochs=10,
    callbacks=callbacks,
)
model = keras.models.load_model("binary_1gram.keras")
print(f"Test acc: {model.evaluate(binary_1gram_test_ds)[1]:.3f}")


# #### Bigrams with binary encoding
# **Configuring the `TextVectorization` layer to return bigrams**
text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="multi_hot",
)


# **Training and testing the binary bigram model**
text_vectorization.adapt(text_only_train_ds)
binary_2gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
binary_2gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
binary_2gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)

model = get_model()
model.summary()
callbacks = [keras.callbacks.ModelCheckpoint("binary_2gram.keras", save_best_only=True)]
model.fit(
    binary_2gram_train_ds.cache(),
    validation_data=binary_2gram_val_ds.cache(),
    epochs=10,
    callbacks=callbacks,
)
model = keras.models.load_model("binary_2gram.keras")
print(f"Test acc: {model.evaluate(binary_2gram_test_ds)[1]:.3f}")


# #### Bigrams with TF-IDF encoding
# **Configuring the `TextVectorization` layer to return token counts**
text_vectorization = TextVectorization(ngrams=2, max_tokens=20000, output_mode="count")


# **Configuring `TextVectorization` to return TF-IDF-weighted outputs**
text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="tf_idf",
)


# **Training and testing the TF-IDF bigram model**
text_vectorization.adapt(text_only_train_ds)

tfidf_2gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
tfidf_2gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
tfidf_2gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)

model = get_model()
model.summary()
callbacks = [keras.callbacks.ModelCheckpoint("tfidf_2gram.keras", save_best_only=True)]
model.fit(
    tfidf_2gram_train_ds.cache(),
    validation_data=tfidf_2gram_val_ds.cache(),
    epochs=10,
    callbacks=callbacks,
)
model = keras.models.load_model("tfidf_2gram.keras")
print(f"Test acc: {model.evaluate(tfidf_2gram_test_ds)[1]:.3f}")


inputs = keras.Input(shape=(1,), dtype="string")
processed_inputs = text_vectorization(inputs)
outputs = model(processed_inputs)
inference_model = keras.Model(inputs, outputs)


import tensorflow as tf

raw_text_data = tf.convert_to_tensor(
    [
        ["That was an excellent movie, I loved it."],
    ]
)
predictions = inference_model(raw_text_data)
print(f"{float(predictions[0] * 100):.2f} percent positive")
