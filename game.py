import sys
import os
import numpy as np
import tensorflow as tf
import argparse
import yaml
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
from collections import namedtuple
from agent import Agents
import env

def shuffle_image_activations(im_acts):
    reordering = np.array(range(len(im_acts)))
    random.shuffle(reordering)
    target_ind = np.argmin(reordering)
    shuffled = im_acts[reordering]
    return (shuffled, target_ind)

def run_game(config, noise_probability):
    image_embedding_dim = config['image_embedding_dim']
    word_embedding_dim = config['word_embedding_dim']
    data_dir = config['data_dir']
    image_dirs = config['image_dirs']
    vocab = config['vocab']
    log_path = config['log_path']
    model_weights = config['model_weights']
    learning_rate = config['learning_rate']
    load_model = config['load_model'] == 'True'

    iterations = config['iterations']
    mini_batch_size = config['mini_batch_size']
    
    agents = Agents(vocab, 
                    image_embedding_dim,
                    word_embedding_dim,
                    learning_rate,
                    temperature=10,
                    batch_size=2,
                    noise_probability=noise_probability)

    environ = env.Environment()
    model = VGG16()

    Game = namedtuple("Game", ["im_acts", "target_acts", "distractor_acts", "word_probs",
                               "image_probs", "target", "word", "selection", "reward"])
    
    batch = []
    tot_reward = 0
    success_count = 0
    success_rates = []
    cumulative_rewards = []
    actual_labels = []
    predicted_labels = []

    start_time = time.time()

    for i in range(iterations):
        writer = tf.summary.create_file_writer(log_path)
        with writer.as_default():
            tf.summary.scalar('Cumulative Reward', tot_reward, step=i)
            writer.flush()
            print('Episode {}/{}'.format(i, iterations), end='\r')

        target_image, distractor_image = environ.get_images()
        target_image = target_image.reshape((1, 224, 224, 3))
        distractor_image = distractor_image.reshape((1, 224, 224, 3))

        target_class = environ.target_class

        td_images = np.vstack([target_image, distractor_image])
        td_acts = model.predict(td_images)

        target_acts = td_acts[0].reshape((1, 1000))
        distractor_acts = td_acts[1].reshape((1, 1000))

        word_probs, word_selected = agents.get_sender_word_probs(target_acts, distractor_acts)

        reordering = np.array([0,1])
        random.shuffle(reordering)
        target = np.where(reordering==0)[0][0]

        img_array = [target_acts, distractor_acts]
        im1_acts, im2_acts = [img_array[reordering[i]] for i in range(len(img_array))]

        receiver_probs, image_selected = agents.get_receiver_selection(word_selected, im1_acts, im2_acts)

        reward = 0.0
        if target == image_selected:
            reward = 1.0
            success_count += 1

        shuffled_acts = np.concatenate([im1_acts, im2_acts])

        batch.append(Game(shuffled_acts, target_acts, distractor_acts,
                          word_probs, receiver_probs, target, word_selected, image_selected, reward))

        if (i+1) % mini_batch_size == 0:
            print('Updating the agent weights')
            agents.update(batch)
            batch = []

        tot_reward += reward

        # Update success rates and cumulative rewards for plotting
        success_rates.append(success_count / (i + 1))
        cumulative_rewards.append(tot_reward)

        # Record actual and predicted labels for the confusion matrix
        actual_labels.append(str(target_class))
        predicted_labels.append(image_selected)  # assuming image_selected is an integer

        print(target_class, reward)

    learning_time = time.time() - start_time
    print(f"\nFinal Success Rate: {success_count / iterations}, Learning Time: {learning_time}")

    # Assuming target_class is a string and image_selected is an integer index
    label_mapping = {'cat': 0, 'dog': 1}
    mapped_actual_labels = [label_mapping[label] for label in actual_labels]
    mapped_predicted_labels = predicted_labels  # predicted_labels should already be integers

    return success_rates, cumulative_rewards, mapped_actual_labels, mapped_predicted_labels

def plot_confusion_matrix(actual, predicted, title):
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def plot_success_rates(sr_without_noise, sr_with_noise):
    plt.figure(figsize=(12, 6))
    plt.plot(sr_without_noise, label='Success Rate without Noise')
    plt.plot(sr_with_noise, label='Success Rate with Noise')
    plt.xlabel('Iterations')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Over Iterations')
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.conf) as g:
        config = yaml.load(g, Loader=yaml.SafeLoader)

    # Run game without noise
    sr_without_noise, cr_without_noise, actual_without_noise, predicted_without_noise = run_game(config, noise_probability=0)
    
    # Run game with noise
    sr_with_noise, cr_with_noise, actual_with_noise, predicted_with_noise = run_game(config, noise_probability=config.get('noise_probability', 0.1))

    # Plot the results for success rates
    plot_success_rates(sr_without_noise, sr_with_noise)

    # Plot confusion matrices
    plot_confusion_matrix(actual_without_noise, predicted_without_noise, 'Confusion Matrix without Noise')
    plot_confusion_matrix(actual_with_noise, predicted_with_noise, 'Confusion Matrix with Noise')

if __name__ == '__main__':
    main()
