# silence tf warnings
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from keras.layers import StringLookup
import numpy as np
from numpy import ndarray


max_len = 21
# Mapping characters to integers.
characters = [
	'!', '"', '#', '&', "'", '(', ')', '*', '+', ',',
	'-', '.', '/', '0', '1', '2', '3', '4', '5', '6',
	'7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D',
	'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
	'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
	'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
	'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
	's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

class ModelInterface:
	def __init__(self, path):
		self.model = tf.keras.models.load_model(path, compile=False)
		self.model.compile()

	@staticmethod
	def img2tensor(image: ndarray):
		return tf.expand_dims(image, 0)

class Classifier(ModelInterface):
	def classify(self, image: ndarray):
		image_tensor = self.img2tensor(image)
		pred = self.model(image_tensor)
		input_len = np.ones(pred.shape[0]) * pred.shape[1]
		# Use greedy search. For complex tasks, you can use beam search.
		results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
			:, :max_len
		]
		# Iterate over the results and get back the text.
		output_text = []
		for res in results:
			print(res)
			res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
			res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
			output_text.append(res)
		return output_text

