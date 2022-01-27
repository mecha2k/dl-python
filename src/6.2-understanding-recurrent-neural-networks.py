from tensorflow import keras

print(keras.__version__)

# # Understanding recurrent neural networks#
# ## A first recurrent layer in Keras
# The process we just naively implemented in Numpy corresponds to an actual Keras layer: the `SimpleRNN` layer:

from keras.layers import SimpleRNN


# There is just one minor difference: `SimpleRNN` processes batches of sequences, like all other Keras layers, not just
# a single sequence like in our Numpy example. This means that it takes inputs of shape `(batch_size, timesteps,
# input_features)`, rather than `(timesteps,  input_features)`.
#
# Like all recurrent layers in Keras, `SimpleRNN` can be run in two different modes: it can return either the full
# sequences of successive outputs for each timestep (a 3D tensor of shape `(batch_size, timesteps, output_features)`),
# or it can return only the last output for each input sequence (a 2D tensor of shape `(batch_size, output_features)`).
# These two modes are controlled by the `return_sequences` constructor argument. Let's take a look at an example:

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()


# It is sometimes useful to stack several recurrent layers one after the other in order to increase the representational
# power of a network. In such a setup, you have to get all intermediate layers to return full sequences:

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # This last layer only returns the last outputs.
model.summary()


# Now let's try to use such a model on the IMDB movie review classification problem. First, let's preprocess the data:

from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print("Loading data...")
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), "train sequences")
print(len(input_test), "test sequences")

print("Pad sequences (samples x time)")
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print("input_train shape:", input_train.shape)
print("input_test shape:", input_test.shape)


# Let's train a simple recurrent network using an `Embedding` layer and a `SimpleRNN` layer:

from keras.layers import Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(input_train, y_train, epochs=1, batch_size=128, validation_split=0.2)


# Let's display the training and validation loss and accuracy:

import matplotlib.pyplot as plt

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig("images/06-02-01")

plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig("images/06-02-02")

# As a reminder, in chapter 3, our very first naive approach to this very dataset got us to 88% test accuracy.
# Unfortunately, our small recurrent network doesn't perform very well at all compared to this baseline (only up to 85%
# validation accuracy). Part of the problem is that our inputs only consider the first 500 words rather the full
# sequences -- hence our RNN has access to less information than our earlier baseline model. The remainder of the
# problem is simply that `SimpleRNN` isn't very good at processing long sequences, like text. Other types of recurrent
# layers perform much better. Let's take a look at some more advanced layers.


# ## A concrete LSTM example in Keras
#
# Now let's switch to more practical concerns: we will set up a model using a LSTM layer and train it on the IMDB data.
# Here's the network, similar to the one with `SimpleRNN` that we just presented. We only specify the output
# dimensionality of the LSTM layer, and leave every other argument (there are lots) to the Keras defaults. Keras has
# good defaults, and things will almost always "just work" without you having to spend time tuning parameters by hand.

from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(input_train, y_train, epochs=1, batch_size=128, validation_split=0.2)

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))
plt.figure()
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig("images/06-02-03")

plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig("images/06-02-04")
