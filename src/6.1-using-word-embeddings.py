from tensorflow import keras

from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import os

print(keras.__version__)

imdb_dir = "data/aclImdb"
train_dir = os.path.join(imdb_dir, "train")

texts, labels = [], []
for label_type in ["neg", "pos"]:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == ".txt":
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == "neg":
                labels.append(0)
            else:
                labels.append(1)


def word_embedding_nn():
    # Number of words to consider as features
    max_features = 10000
    # Cut texts after this number of words
    # (among top max_features most common words)
    maxlen = 20

    # Load the data as lists of integers.
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    # This turns our lists of integers
    # into a 2D integer tensor of shape `(samples, maxlen)`
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    print(x_train.shape)
    print(x_train[0])

    model = Sequential()
    # We specify the maximum input length to our Embedding layer, so we can later flatten the embedded inputs
    model.add(Embedding(input_dim=10000, output_dim=8, input_length=maxlen))
    # After the Embedding layer, our activations have shape `(samples, maxlen, 8)`.
    # We flatten the 3D tensor of embeddings into a 2D tensor of shape `(samples, maxlen * 8)`
    model.add(Flatten())

    # We add the classifier on top
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    model.summary()

    model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2)


def word_embedding_pre_trained(texts, labels):
    maxlen = 100  # We will cut reviews after 100 words
    training_samples = 200  # We will be training on 200 samples
    validation_samples = 10000  # We will be validating on 10000 samples
    max_words = 10000  # We will only consider the top 10,000 words in the dataset

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print("Found %s unique tokens." % len(word_index))

    data = pad_sequences(sequences, maxlen=maxlen)

    labels = np.asarray(labels)
    print("Shape of data tensor:", data.shape)
    print("Shape of label tensor:", labels.shape)

    # Split the data into a training set and a validation set
    # But first, shuffle the data, since we started from data
    # where sample are ordered (all negative first, then all positive).
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples : training_samples + validation_samples]
    y_val = labels[training_samples : training_samples + validation_samples]

    # ### Download the GloVe word embeddings
    # Head to `https://nlp.stanford.edu/projects/glove/` (where you can learn more about the GloVe algorithm), and download
    # the pre-computed embeddings from 2014 English Wikipedia. It's a 822MB zip file named `glove.6B.zip`, containing
    # 100-dimensional embedding vectors for 400,000 words (or non-word tokens). Un-zip it.

    # ### Pre-process the embeddings
    # Let's parse the unzipped file (it's a `txt` file) to build an index mapping words (as strings) to their vector
    # representation (as number vectors).

    glove_dir = "data/glove.6B"
    embeddings_index = {}
    f = open(os.path.join(glove_dir, "glove.6B.100d.txt"))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()
    print("Found %s word vectors." % len(embeddings_index))

    # Now let's build an embedding matrix that we will be able to load into an `Embedding` layer. It must be a matrix of
    # shape `(max_words, embedding_dim)`, where each entry `i` contains the `embedding_dim`-dimensional vector for the word
    # of index `i` in our reference word index (built during tokenization). Note that the index `0` is not supposed to
    # stand for any word or token -- it's a placeholder.

    embedding_dim = 100

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

    # ### Define a model
    # We will be using the same model architecture as before:

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()

    # ### Load the GloVe embeddings in the model
    # The `Embedding` layer has a single weight matrix: a 2D float matrix where each entry `i` is the word vector meant to
    # be associated with index `i`. Simple enough. Let's just load the GloVe matrix we prepared into our `Embedding` layer,
    # the first layer in our model:

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    # Additionally, we freeze the embedding layer (we set its `trainable` attribute to `False`), following the same
    # rationale as what you are already familiar with in the context of pre-trained convnet features: when parts of a model
    # are pre-trained (like our `Embedding` layer), and parts are randomly initialized (like our classifier), the pre-trained
    # parts should not be updated during training to avoid forgetting what they already know. The large gradient update
    # triggered by the randomly initialized layers would be very disruptive to the already learned features.

    # ### Train and evaluate
    # Let's compile our model and train it:
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
    model.save_weights("data/pre_trained_glove_model.h5")

    # Let's plot its performance over time:
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.savefig("images/06-01-01")

    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig("images/06-01-02")

    # The model quickly starts overfitting, unsurprisingly given the small number of training samples. Validation accuracy
    # has high variance for the same reason, but seems to reach high 50s.
    #
    # Note that your mileage may vary: since we have so few training samples, performance is heavily dependent on which
    # exact 200 samples we picked, and we picked them at random. If it worked really poorly for you, try picking a different
    # random set of 200 samples, just for the sake of the exercise (in real life you don't get to pick your training data).
    #
    # We can also try to train the same model without loading the pre-trained word embeddings and without freezing the
    # embedding layer. In that case, we would be learning a task-specific embedding of our input tokens, which is generally
    # more powerful than pre-trained word embeddings when lots of data is available. However, in our case, we have only 200
    # training samples. Let's try it:

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.savefig("images/06-01-03")

    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig("images/06-01-04")

    # Validation accuracy stalls in the low 50s. So in our case, pre-trained word embeddings does outperform jointly learned
    # embeddings. If you increase the number of training samples, this will quickly stop being the case, try it as an exercise.
    #
    # Finally, let's evaluate the model on the test data. First, we will need to tokenize the test data:
    test_dir = os.path.join(imdb_dir, "test")
    texts, labels = [], []
    for label_type in ["neg", "pos"]:
        dir_name = os.path.join(test_dir, label_type)
        for fname in sorted(os.listdir(dir_name)):
            if fname[-4:] == ".txt":
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == "neg":
                    labels.append(0)
                else:
                    labels.append(1)

    sequences = tokenizer.texts_to_sequences(texts)
    x_test = pad_sequences(sequences, maxlen=maxlen)
    y_test = np.asarray(labels)

    # And let's load and evaluate the first model:
    model.load_weights("data/pre_trained_glove_model.h5")
    model.evaluate(x_test, y_test)


if __name__ == "__main__":
    word_embedding_pre_trained(texts, labels)

# We get an appalling test accuracy of 54%. Working with just a handful of training samples is hard!
