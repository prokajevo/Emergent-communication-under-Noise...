import tensorflow as tf

# Example of a simplified receiver model using standard Keras layers
receiver_model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(1000,))  # Adjust input shape as needed
])

# Test forward pass
test_input = tf.random.normal([1, 1000])
with tf.GradientTape() as tape:
    output = receiver_model(test_input)
    loss = tf.reduce_mean(output)  # Use a simple loss for testing

# Check gradients
gradients = tape.gradient(loss, receiver_model.trainable_variables)
print("Gradients:", gradients)

"""import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.optimizers import Adam


class LinearSigmoid(Layer):
    def __init__(self, input_dim=32, h_units=32):
        super(LinearSigmoid, self).__init__()
        w_init = tf.random_normal_initializer(stddev=0.01)
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, h_units), dtype='float32'),
            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(h_units,), dtype='float32'),
            trainable=True)

    def call(self, inputs):
        return tf.sigmoid(tf.matmul(inputs, self.w) + self.b)


class Agents:
    def __init__(self, vocab, image_embedding_dim, word_embedding_dim, learning_rate, temperature=10, batch_size=32):
        self.vocab = vocab
        self.image_embedding_dim = image_embedding_dim
        self.batch_size = batch_size
        self.word_embedding_dim = word_embedding_dim
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.vocab_len = len(self.vocab)
        self.build_sender_receiver_model()
        self.build_word_probs_model()
        self.sender_optimizer = Adam(self.learning_rate)
        self.receiver_optimizer = Adam(self.learning_rate)
        w_init = tf.random_normal_initializer(stddev=0.01)
        self.vocab_embedding = tf.Variable(
            w_init(shape=(self.vocab_len, self.word_embedding_dim), dtype='float32'),
            trainable=True)

    def build_word_probs_model(self):
        self.word_probs_model = tf.keras.Sequential()
        self.word_probs_model.add(Dense(self.vocab_len, use_bias=True,
                                        input_shape=(2*self.image_embedding_dim,),
                                        activation='softmax'))

    def build_sender_receiver_model(self):
        self.sender = LinearSigmoid(1000, self.image_embedding_dim)
        self.receiver = LinearSigmoid(1000, self.image_embedding_dim)

    def get_sender_word_probs(self, target_acts, distractor_acts):
        t_embed = self.sender(target_acts)
        d_embed = self.sender(distractor_acts)
        ordered_embed = tf.concat([t_embed, d_embed], axis=1)
        word_probs = self.word_probs_model(ordered_embed)
        word_selected = np.random.choice(np.arange(self.vocab_len), p=word_probs.numpy()[0])
        return word_probs, word_selected

    def get_receiver_selection(self, word, im1_acts, im2_acts):
        word_embed = tf.gather(self.vocab_embedding, word)
        im1_embed = self.receiver(im1_acts)
        im2_embed = self.receiver(im2_acts)
        im1_score = tf.reduce_sum(im1_embed * word_embed, axis=1)
        im2_score = tf.reduce_sum(im2_embed * word_embed, axis=1)
        image_probs = tf.nn.softmax([im1_score, im2_score], axis=0)
        image_selected = np.random.choice(np.arange(2), p=image_probs.numpy().squeeze())  # Use squeeze() here
        return image_probs, image_selected

    def update(self, batch):
        acts, target_acts, distractor_acts, word_probs, receiver_probs, target, word, selection, reward = map(
            lambda x: np.squeeze(np.array(x)), zip(*batch))

        with tf.GradientTape(persistent=True) as tape:
            # Assuming that 'word_probs' and 'receiver_probs' are the probabilities associated with the selected actions
            sender_loss = -tf.reduce_mean(tf.math.log(word_probs) * reward)
            receiver_loss = -tf.reduce_mean(tf.math.log(receiver_probs) * reward)

        sender_gradients = tape.gradient(sender_loss, self.sender.trainable_variables)
        receiver_gradients = tape.gradient(receiver_loss, self.receiver.trainable_variables)

        self.sender_optimizer.apply_gradients(zip(sender_gradients, self.sender.trainable_variables))
        self.receiver_optimizer.apply_gradients(zip(receiver_gradients, self.receiver.trainable_variables))
        del tape  # Drop the reference to the tape


if __name__ == '__main__':
    vocab = ['dog', 'cat', 'mouse']
    agent = Agents(vocab=vocab, image_embedding_dim=10, word_embedding_dim=10, learning_rate=0.001)
    # Example usage: Call agent.update(...) with a batch of experiences"""
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.optimizers.legacy import Adam
import random

class LinearSigmoid(Layer):
    def __init__(self, input_dim=32, h_units=32):
        super(LinearSigmoid, self).__init__()
        w_init = tf.random_normal_initializer(stddev=0.01)
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, h_units), dtype='float32'),
            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(h_units,), dtype='float32'),
            trainable=True)

    def call(self, inputs):
        return tf.sigmoid(tf.matmul(inputs, self.w) + self.b)

class Agents:
    def __init__(self, vocab, image_embedding_dim, word_embedding_dim,
                 learning_rate, temperature=10, batch_size=32, noise_probability=0.1):
        self.vocab = vocab
        self.image_embedding_dim = image_embedding_dim
        self.batch_size = batch_size
        self.word_embedding_dim = word_embedding_dim
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.noise_probaility = noise_probability
        self.vocab_len = len(self.vocab)
        self.build_sender_receiver_model()
        self.build_word_probs_model()
        self.sender_optimizer = Adam(self.learning_rate)
        self.receiver_optimizer = Adam(self.learning_rate)
        w_init = tf.random_normal_initializer(stddev=0.01)
        self.vocab_embedding = tf.Variable(
            w_init(shape=(self.vocab_len, self.word_embedding_dim), dtype='float32'),
            trainable=True)

    def build_word_probs_model(self):
        self.word_probs_model = tf.keras.Sequential()
        self.word_probs_model.add(Dense(2, use_bias=True,
                                        input_shape=(None, 2*self.image_embedding_dim),
                                        activation='softmax'))

    def get_receiver_selection(self, word, im1_acts, im2_acts):
        word_embed = tf.squeeze(tf.gather(self.vocab_embedding, self.word))
        im1_embed = self.receiver(im1_acts)
        im2_embed = self.receiver(im2_acts)
        im1_score = tf.reduce_sum(tf.multiply(im1_embed, word_embed), axis=1).numpy()[0]
        im2_score = tf.reduce_sum(tf.multiply(im2_embed, word_embed), axis=1).numpy()[0]
        image_probs = tf.nn.softmax([im1_score, im2_score]).numpy()
        selection = np.random.choice(np.arange(2), p=image_probs)
        return image_probs, selection

    def build_sender_receiver_model(self):
        self.sender = LinearSigmoid(1000, self.image_embedding_dim)
        self.receiver = LinearSigmoid(1000, self.image_embedding_dim)

    def get_sender_word_probs(self, target_acts, distractor_acts, noise_probability=0.1):
        t_embed = self.sender(target_acts)
        d_embed = self.sender(distractor_acts)
        ordered_embed = tf.concat([t_embed, d_embed], axis=1)
        self.word_probs = self.word_probs_model(ordered_embed).numpy()[0]
        self.word = np.random.choice(np.arange(len(self.vocab)), p=self.word_probs)

        if random.random() < noise_probability:
            self.word = 1 - self.word
            print('Here is a Noise')
        
        return self.word_probs, self.word

    def update(self, batch):
        # Unpack the batch data
        acts, target_acts, distractor_acts, word_probs, \
            receiver_probs, target, word, selection, reward = map(lambda x: np.squeeze(np.array(x)), zip(*batch))

        # Reshape the data as needed
        reward = np.reshape(reward, [-1, 1])
        selection = np.reshape(selection, [1, -1])
        word = np.reshape(word, [1, -1])
        target_acts = np.reshape(target_acts, [-1, 1000])
        distractor_acts = np.reshape(distractor_acts, [-1, 1000])
        acts = np.reshape(acts, [-1, 2000])
        receiver_probs = np.reshape(receiver_probs, [-1, 2])

        # Calculate the losses
        sender_loss = self.calculate_sender_loss(word_probs, word, reward)
        receiver_loss = self.calculate_receiver_loss(receiver_probs, selection, reward)

        # Perform gradient updates
        with tf.GradientTape() as tape_sender:
            sender_gradients = tape_sender.gradient(sender_loss, self.sender.trainable_variables)
            self.sender_optimizer.apply_gradients(zip(sender_gradients, self.sender.trainable_variables))

        with tf.GradientTape() as tape_receiver:
            receiver_gradients = tape_receiver.gradient(receiver_loss, self.receiver.trainable_variables)
            self.receiver_optimizer.apply_gradients(zip(receiver_gradients, self.receiver.trainable_variables))

    def calculate_sender_loss(self, word_probs, word, reward):
       loss = tf.reduce_mean(-tf.reduce_sum(tf.math.log(word_probs) * word, axis=1) * reward)
       return loss
        

    def calculate_receiver_loss(self, receiver_probs, selection, reward):
        loss = tf.reduce_mean(-tf.reduce_sum(tf.math.log(receiver_probs) * selection, axis=1) * reward)
        return loss

if __name__=='__main__':
    vocab = ['dog', 'cat', 'mouse']
    agent = Agents(vocab=vocab, image_embedding_dim=10, word_embedding_dim=10,
                   learning_rate=0.2, temperature=10, batch_size=32)
    t_acts = tf.ones((1, 1000))
    d_acts = tf.ones((1, 1000))
    word_probs = agent.get_sender_word_probs(t_acts, d_acts)
    """