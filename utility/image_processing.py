import os
from pathlib import Path
from random import shuffle
import numpy as np
from skimage.data import imread
from skimage.transform import resize
from skimage.io import imsave
import tensorflow as tf

from utility.debug_tools import pwc, assert_colorize
import utility.utils as utils

def read_image(image_path, image_shape=None, preserve_range=True):
    image = imread(image_path)
    if image_shape:
        image = resize(image, image_shape, preserve_range=preserve_range)
    image = np.expand_dims(image, 0)

    return image

def norm_image(image, norm_range=[0, 1]):
    if norm_range == [0, 1]:
        return image / 255.0
    elif norm_range == [-1, 1]:
        return image / 127.5 - 1.
    else:
        raise NotImplementedError

def save_image(images, path, size=None):
    assert_colorize(len(images.shape) == 4, f'images should be 4D, but get shape {images.shape}')
    num_images = images.shape[0]
    if size is None:
        size = utils.squarest_grid_size(num_images)
    images = merge(images, size)
    utils.check_make_dir(path)
    imsave(path, images)

def merge(images, size):
    assert_colorize(len(images.shape) == 4, f'images should be 4D, but get shape {images.shape}')
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        NotImplementedError

def image_dataset(ds_dir, image_size, batch_size, norm_range=None):
    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        if norm_range:
            image = norm_image(image, norm_range)
        return image

    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        return preprocess_image(image)

    ds_dir = Path(ds_dir)
    assert_colorize(ds_dir.is_dir(), f'Not a valid directory {ds_dir}')
    all_image_paths = [str(f) for f in Path(ds_dir).glob('*')]
    pwc(f'Total Images: {len(all_image_paths)}', 'magenta')
    ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    ds = ds.shuffle(buffer_size = len(all_image_paths))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    image = ds.make_one_shot_iterator().get_next('images')

    return ds, image

class ImageGenerator:
    def __init__(self, ds_dir, image_shape, batch_size, preserve_range=True):
        self.all_image_paths = [str(f) for f in Path(ds_dir).glob('*')]
        pwc(f'Total Images: {len(self.all_image_paths)}', 'magenta')
        self.total_images = len(self.all_image_paths)
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.preserve_range = preserve_range
        self.idx = 0

    def __call__(self):
        while True:
            yield self.sample()
    
    def sample(self):
        if self.idx == 0:
            shuffle(self.all_image_paths)
        
        batch_path = self.all_image_paths[self.idx: self.idx + self.batch_size]
        batch_image = [imread(path) for path in batch_path]
        batch_image = np.array([resize(img, self.image_shape, preserve_range=self.preserve_range) for img in batch_image], dtype=np.float32)
        self.idx += self.batch_size
        if self.idx >= self.total_images:
            self.idx = 0

        return batch_image
    