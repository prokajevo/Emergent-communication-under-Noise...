import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

class Environment:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.ds_train, self.ds_info = tfds.load(
            'cats_vs_dogs',
            split='train[:80%]',
            as_supervised=True,
            with_info=True,
        )
        self.ds_train = self.ds_train.map(self.preprocess).batch(1).prefetch(tf.data.AUTOTUNE)
        self.iterator = iter(self.ds_train)

    def preprocess(self, image, label):
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def get_images(self):
        # Get a pair of images
        image1, label1 = next(self.iterator)
        image2, label2 = next(self.iterator)

        # Ensure the two images are not from the same class
        while label1.numpy() == label2.numpy():
            image2, label2 = next(self.iterator)

        self.target = image1.numpy().squeeze()
        self.distractor = image2.numpy().squeeze()
        self.target_class = label1.numpy()

        return self.target, self.distractor

    def reset(self):
        self.iterator = iter(self.ds_train)