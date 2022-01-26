#!/usr/bin/env python
# coding: utf-8

# This is a companion notebook for the book [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
# 
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
# 
# This notebook was generated for TensorFlow 2.6.

# # The mathematical building blocks of neural networks

# ## A first look at a neural network

# **Loading the MNIST dataset in Keras**

# In[ ]:


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[ ]:


train_images.shape


# In[ ]:


len(train_labels)


# In[ ]:


train_labels


# In[ ]:


test_images.shape


# In[ ]:


len(test_labels)


# In[ ]:


test_labels


# **The network architecture**

# In[ ]:


from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])


# **The compilation step**

# In[ ]:


model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# **Preparing the image data**

# In[ ]:


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255


# **"Fitting" the model**

# In[ ]:


model.fit(train_images, train_labels, epochs=5, batch_size=128)


# **Using the model to make predictions**

# In[ ]:


test_digits = test_images[0:10]
predictions = model.predict(test_digits)
predictions[0]


# In[ ]:


predictions[0].argmax()


# In[ ]:


predictions[0][7]


# In[ ]:


test_labels[0]


# **Evaluating the model on new data**

# In[ ]:


test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")


# ## Data representations for neural networks

# ### Scalars (rank-0 tensors)

# In[ ]:


import numpy as np
x = np.array(12)
x


# In[ ]:


x.ndim


# ### Vectors (rank-1 tensors)

# In[ ]:


x = np.array([12, 3, 6, 14, 7])
x


# In[ ]:


x.ndim


# ### Matrices (rank-2 tensors)

# In[ ]:


x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
x.ndim


# ### Rank-3 and higher-rank tensors

# In[ ]:


x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])
x.ndim


# ### Key attributes

# In[ ]:


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[ ]:


train_images.ndim


# In[ ]:


train_images.shape


# In[ ]:


train_images.dtype


# **Displaying the fourth digit**

# In[ ]:


import matplotlib.pyplot as plt
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


# In[ ]:


train_labels[4]


# ### Manipulating tensors in NumPy

# In[ ]:


my_slice = train_images[10:100]
my_slice.shape


# In[ ]:


my_slice = train_images[10:100, :, :]
my_slice.shape


# In[ ]:


my_slice = train_images[10:100, 0:28, 0:28]
my_slice.shape


# In[ ]:


my_slice = train_images[:, 14:, 14:]


# In[ ]:


my_slice = train_images[:, 7:-7, 7:-7]


# ### The notion of data batches

# In[ ]:


batch = train_images[:128]


# In[ ]:


batch = train_images[128:256]


# In[ ]:


n = 3
batch = train_images[128 * n:128 * (n + 1)]


# ### Real-world examples of data tensors

# ### Vector data

# ### Timeseries data or sequence data

# ### Image data

# ### Video data

# ## The gears of neural networks: tensor operations

# ### Element-wise operations

# In[ ]:


def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


# In[ ]:


def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


# In[ ]:


import time

x = np.random.random((20, 100))
y = np.random.random((20, 100))

t0 = time.time()
for _ in range(1000):
    z = x + y
    z = np.maximum(z, 0.)
print("Took: {0:.2f} s".format(time.time() - t0))


# In[ ]:


t0 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(z)
print("Took: {0:.2f} s".format(time.time() - t0))


# ### Broadcasting

# In[ ]:


import numpy as np
X = np.random.random((32, 10))
y = np.random.random((10,))


# In[ ]:


y = np.expand_dims(y, axis=0)


# In[ ]:


Y = np.concatenate([y] * 32, axis=0)


# In[ ]:


def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


# In[ ]:


import numpy as np
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x, y)


# ### Tensor product

# In[ ]:


x = np.random.random((32,))
y = np.random.random((32,))
z = np.dot(x, y)


# In[ ]:


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


# In[ ]:


def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


# In[ ]:


def naive_matrix_vector_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z


# In[ ]:


def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z


# ### Tensor reshaping

# In[ ]:


train_images = train_images.reshape((60000, 28 * 28))


# In[ ]:


x = np.array([[0., 1.],
             [2., 3.],
             [4., 5.]])
x.shape


# In[ ]:


x = x.reshape((6, 1))
x


# In[ ]:


x = np.zeros((300, 20))
x = np.transpose(x)
x.shape


# ### Geometric interpretation of tensor operations

# ### A geometric interpretation of deep learning

# ## The engine of neural networks: gradient-based optimization

# ### What's a derivative?

# ### Derivative of a tensor operation: the gradient

# ### Stochastic gradient descent

# ### Chaining derivatives: The Backpropagation algorithm

# #### The chain rule

# #### Automatic differentiation with computation graphs

# #### The gradient tape in TensorFlow

# In[ ]:


import tensorflow as tf
x = tf.Variable(0.)
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)


# In[ ]:


x = tf.Variable(tf.random.uniform((2, 2)))
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)


# In[ ]:


W = tf.Variable(tf.random.uniform((2, 2)))
b = tf.Variable(tf.zeros((2,)))
x = tf.random.uniform((2, 2))
with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + b
grad_of_y_wrt_W_and_b = tape.gradient(y, [W, b])


# ## Looking back at our first example

# In[ ]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255


# In[ ]:


model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])


# In[ ]:


model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# In[ ]:


model.fit(train_images, train_labels, epochs=5, batch_size=128)


# ### Reimplementing our first example from scratch in TensorFlow

# #### A simple Dense class

# In[ ]:


import tensorflow as tf

class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]


# #### A simple Sequential class

# In[ ]:


class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
           x = layer(x)
        return x

    @property
    def weights(self):
       weights = []
       for layer in self.layers:
           weights += layer.weights
       return weights


# In[ ]:


model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])
assert len(model.weights) == 4


# #### A batch generator

# In[ ]:


import math

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


# ### Running one training step

# In[ ]:


def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


# In[ ]:


learning_rate = 1e-3

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate)


# In[ ]:


from tensorflow.keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))


# ### The full training loop

# In[ ]:


def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")


# In[ ]:


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128)


# ### Evaluating the model

# In[ ]:


predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")


# ## Summary
