import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
from skimage import morphology
from contextlib import contextmanager
import cv2


image_size = (128, 32) # WxH
DESTDIR = Path(os.getcwd()) / "preprocessing"
Path.mkdir(DESTDIR, parents=True, exist_ok=True)

@contextmanager
def window(fn):
	try:
		yield
	except Exception as e:
		print(f"Error encountered in {fn}\n{e}")
		exit(1)

def store_transformation(transformation):
	def inner(x):
		assert x.shape[:2][::-1] == image_size, x.shape[:2][::-1]
		with window(transformation.__name__):
			tr = transformation(x)
			keras.preprocessing.image.save_img(
				str(DESTDIR / transformation.__name__) + ".png",
				tf.reshape(tr, (*image_size[::-1], 1))
			)
		return tr
	return inner

def image_conversion(func):
	def converted(x):
		with window(func.__name__):
			x = keras.preprocessing.image.array_to_img(x)
			y = func(x)
			y = keras.preprocessing.image.img_to_array(y)
		return y
	converted.__name__ = func.__name__
	return converted

def input(image: Image):
	image.save(DESTDIR / "input.png")
	return keras.preprocessing.image.img_to_array(image)

@image_conversion
def resize(image, size=image_size):
	return image.resize(size)

# @image_conversion
def pad_resize(image, size=image_size):
	image = tf.image.resize(image, size=size[::-1], preserve_aspect_ratio=True)
	padding = np.array(size[::-1]) - np.array(image.shape)[:2]
	W, H = padding // 2
	w, h = padding % 2

	return tf.pad(
		image,
		paddings=[
			[W, W+w],
			[H, H+h],
			[0, 0]
		],
		constant_values=255
	)

@store_transformation
# @image_conversion
def grayscale(image):
	return tf.image.rgb_to_grayscale(image)

@store_transformation
@image_conversion
def invert(image):
	return ImageOps.invert(image)

@store_transformation
def skeletonize(image):
	return morphology.skeletonize(image)

@store_transformation
def dilate(image):
	return cv2.dilate(
		image,
		cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)),
		iterations=1
	)

@store_transformation
def normalize(image):
	return tf.cast(image, tf.float32) / 255.0

@store_transformation
def invertnorm(image):
	return tf.where(image > 200, tf.zeros_like(image), tf.ones_like(image)) # squash and invert

@store_transformation
def blur(image, k=3):
	# probably quite bad
	return cv2.blur(image, (k, k), cv2.BORDER_DEFAULT)

@store_transformation
def reshape(image):
	image = tf.transpose(image, perm=[1, 0, 2])
	image = tf.image.flip_left_right(image)
	return image

def debug(image):
	print(image.shape, type(image))
	return image

def preprocessing_pipeline(image: Image, *transformation_pipeline):
	for transformation in transformation_pipeline:
		image = transformation(image)		
	return image

def preprocess(image: Image):
	return preprocessing_pipeline(
		image,
		input,
		pad_resize,
		grayscale,
		invertnorm,
		reshape,
	)