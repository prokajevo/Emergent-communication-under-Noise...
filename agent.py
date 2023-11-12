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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.optimizers import Adam
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
        self.noise_probability = noise_probability
        self.vocab_len = len(vocab)
        self.build_sender_receiver_model()
        self.build_word_probs_model()
        self.sender_optimizer = Adam(learning_rate)
        self.receiver_optimizer = Adam(learning_rate)
        w_init = tf.random_normal_initializer(stddev=0.01)
        self.vocab_embedding = tf.Variable(
            initial_value=w_init(shape=(self.vocab_len, self.word_embedding_dim), dtype='float32'),
            trainable=True)

    def build_sender_receiver_model(self):
        self.sender = LinearSigmoid(1000, self.image_embedding_dim)
        self.receiver = LinearSigmoid(1000, self.image_embedding_dim)

    def build_word_probs_model(self):
        self.word_probs_model = tf.keras.Sequential()
        self.word_probs_model.add(Dense(self.vocab_len, use_bias=True,
                                        input_shape=(self.image_embedding_dim * 2,),
                                        activation='softmax'))

    def get_sender_word_probs(self, target_acts, distractor_acts):
        t_embed = self.sender(target_acts)
        d_embed = self.sender(distractor_acts)

        # Concatenate the embeddings along the last axis to form a single vector per pair
        # The resulting shape should be (batch_size, 2 * image_embedding_dim)
        ordered_embed = tf.concat([t_embed, d_embed], axis=1)

        # Now ordered_embed should be 2D tensor with shape (batch_size, 2 * image_embedding_dim)
        # The input to word_probs_model must be properly reshaped if it is not
        ordered_embed = tf.reshape(ordered_embed, (-1, self.image_embedding_dim * 2))
        print("ordered_embed shape:", ordered_embed.shape)


        word_probs = self.word_probs_model(ordered_embed)

        # Choose a word based on the probabilities for each item in the batch
        words = tf.random.categorical(tf.math.log(word_probs), 1)[:, 0]

        return word_probs, words

    def get_receiver_selection(self, word, im1_acts, im2_acts):
        word_idx = tf.cast(word, tf.int32)
        word_embed = tf.nn.embedding_lookup(self.vocab_embedding, [word_idx])

        im1_embed = self.receiver(im1_acts)
        im2_embed = self.receiver(im2_acts)

        # Calculate the dot product between the word embedding and each image embedding.
        # The resulting scores should be 1-dimensional (scalars), because we sum over the last axis.
        im1_score = tf.reduce_sum(im1_embed * word_embed, axis=-1)
        im2_score = tf.reduce_sum(im2_embed * word_embed, axis=-1)

        # Ensure the scores are scalars by using tf.squeeze
        im1_score = tf.squeeze(im1_score)
        im2_score = tf.squeeze(im2_score)

        # Stack the two scores to create a 1D tensor with shape [2]
        scores = tf.stack([im1_score, im2_score])

        # Apply softmax to get probabilities, which should now be a 1D array with shape [2]
        image_probs = tf.nn.softmax(scores)

        # Convert image_probs to a numpy array and check if it sums to 1
        image_probs_np = image_probs.numpy()
        if not np.isclose(np.sum(image_probs_np), 1):
            raise ValueError(f"Probabilities do not sum to 1: {image_probs_np}")

        # Use the numpy array for random selection
        selection = np.random.choice(np.arange(2), p=image_probs_np)

        return image_probs_np, selection


    def calculate_sender_loss(self, word_probs, word, reward):
        loss = tf.reduce_mean(-tf.reduce_sum(tf.math.log(word_probs) * tf.one_hot(word, depth=self.vocab_len), axis=1) * reward)
        return loss

    def calculate_receiver_loss(self, receiver_probs, selection, reward):
        # Assuming receiver_probs is a 2D tensor with shape [batch_size, num_classes]
        # and selection is a 1D tensor with shape [batch_size]
        # with each element being the index of the chosen class.

        # The one-hot encoding should have the same depth as the number of classes in receiver_probs
        one_hot_selection = tf.one_hot(selection, depth=tf.shape(receiver_probs)[-1])

        # Calculate the cross-entropy loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_selection, logits=receiver_probs)

        # Multiply by rewards
        weighted_loss = cross_entropy * reward

        # Finally, take the mean over the batch
        loss = tf.reduce_mean(weighted_loss)

        return loss


    """def update(self, batch):
        # Unpack the batch data
        acts, target_acts, distractor_acts, _, \
        _, target, word, selection, reward = map(lambda x: np.squeeze(np.array(x)), zip(*batch))

        # Convert numpy arrays to tensors
        target_acts = tf.convert_to_tensor(target_acts, dtype=tf.float32)
        distractor_acts = tf.convert_to_tensor(distractor_acts, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        word = tf.convert_to_tensor(word, dtype=tf.int32)
        selection = tf.convert_to_tensor(selection, dtype=tf.int32)

        # Perform gradient updates for sender
        with tf.GradientTape() as tape_sender:
            sender_word_probs, _ = self.get_sender_word_probs(target_acts, distractor_acts)
            sender_loss = self.calculate_sender_loss(sender_word_probs, word, reward)
        sender_gradients = tape_sender.gradient(sender_loss, self.sender.trainable_variables)
        self.sender_optimizer.apply_gradients(zip(sender_gradients, self.sender.trainable_variables))

        # Perform gradient updates for receiver
        with tf.GradientTape() as tape_receiver:
            receiver_image_probs, _ = self.get_receiver_selection(word, target_acts, distractor_acts)
            receiver_loss = self.calculate_receiver_loss(receiver_image_probs, selection, reward)
        receiver_gradients = tape_receiver.gradient(receiver_loss, self.receiver.trainable_variables)
        self.receiver_optimizer.apply_gradients(zip(receiver_gradients, self.receiver.trainable_variables))"""
    """def update(self, batch):
        # Unpack the batch data
        im_acts, target_acts, distractor_acts, word_probs, \
        receiver_probs, targets, words, selections, rewards = map(np.array, zip(*batch))

        # Convert numpy arrays to tensors
        target_acts = tf.convert_to_tensor(target_acts, dtype=tf.float32)
        distractor_acts = tf.convert_to_tensor(distractor_acts, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        selections = tf.convert_to_tensor(selections, dtype=tf.int32)
        words = tf.convert_to_tensor(words, dtype=tf.int32)

        # Perform gradient updates for sender
        with tf.GradientTape() as tape_sender:
            sender_word_probs, _ = self.get_sender_word_probs(target_acts, distractor_acts)
            sender_loss = self.calculate_sender_loss(sender_word_probs, words, rewards)
        sender_gradients = tape_sender.gradient(sender_loss, self.sender.trainable_variables)
        self.sender_optimizer.apply_gradients(zip(sender_gradients, self.sender.trainable_variables))

        # Perform gradient updates for receiver
        with tf.GradientTape() as tape_receiver:
            receiver_image_probs, _ = self.get_receiver_selection(words, target_acts, distractor_acts)
            receiver_loss = self.calculate_receiver_loss(receiver_image_probs, selections, rewards)
        receiver_gradients = tape_receiver.gradient(receiver_loss, self.receiver.trainable_variables)
        self.receiver_optimizer.apply_gradients(zip(receiver_gradients, self.receiver.trainable_variables))"""
    def update(self, batch):
        im_acts, target_acts, distractor_acts, word_probs, \
        receiver_probs, targets, words, selections, rewards = map(np.array, zip(*batch))

        # Since 'im_acts' contains both 'im1_acts' and 'im2_acts' concatenated, we need to split them
        # Assuming that 'im_acts' has the shape (batch_size, concatenated_size)
        # and that each 'im1_acts' and 'im2_acts' have half of 'concatenated_size'
        half_size = im_acts.shape[1] // 2
        im1_acts = im_acts[:, :half_size]
        im2_acts = im_acts[:, half_size:]

        # Convert numpy arrays to tensors
        target_acts = tf.convert_to_tensor(target_acts, dtype=tf.float32)
        distractor_acts = tf.convert_to_tensor(distractor_acts, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        selections = tf.convert_to_tensor(selections, dtype=tf.int32)
        words = tf.convert_to_tensor(words, dtype=tf.int32)
        with tf.GradientTape() as tape_sender:
            sender_word_probs, _ = self.get_sender_word_probs(target_acts, distractor_acts)
            sender_loss = self.calculate_sender_loss(sender_word_probs, words, rewards)
        sender_gradients = tape_sender.gradient(sender_loss, self.sender.trainable_variables)
        self.sender_optimizer.apply_gradients(zip(sender_gradients, self.sender.trainable_variables))

        # Perform gradient updates for receiver
        with tf.GradientTape() as tape_receiver:
            # Ensure all operations are being watched
            tape_receiver.watch(self.receiver.trainable_variables)
            print("Receiver trainable variables:", self.receiver.trainable_variables)

            # Compute the receiver's output for all games in the batch
            receiver_probs = []
            for game_data in batch:
                # Unpack game_data for the receiver selection step
                _, im1_act, im2_act, _, _, _, word, _, _ = game_data
                receiver_prob, _ = self.get_receiver_selection(word, im1_act, im2_act)
                receiver_probs.append(receiver_prob)

            # Stack all probabilities and selections for the batch
            receiver_probs = tf.stack(receiver_probs)
            selections = tf.stack([game_data.selection for game_data in batch])
            rewards = tf.stack([game_data.reward for game_data in batch])

            # Compute the loss for the entire batch
            receiver_loss = self.calculate_receiver_loss(receiver_probs, selections, rewards)

        # Compute gradients for receiver and check if they are not None
        receiver_gradients = tape_receiver.gradient(receiver_loss, self.receiver.trainable_variables)
        for var, grad in zip(self.receiver.trainable_variables, receiver_gradients):
            if grad is None:
                print(f"Gradient not computed for: {var.name}")
            else:
                print(f"Variable: {var.name}, Gradient: {grad}")

        # Apply gradients if they are valid
        if all(grad is not None for grad in receiver_gradients):
            self.receiver_optimizer.apply_gradients(zip(receiver_gradients, self.receiver.trainable_variables))
        else:
            raise ValueError("One or more gradients are None for receiver variables.")