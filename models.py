# silence tf warnings
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import ndarray


max_len = 21
# Mapping characters to integers.
char2num = list("!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

def decode_batch_predictions(pred):
	input_len = np.ones(pred.shape[0]) * pred.shape[1]
	# Use greedy search. For complex tasks, you can use beam search.
	results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
		:, :max_len
	]
	# Iterate over the results and get back the text.
	output_text = []
	for res in results:
		res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
		res = "".join((char2num[(x-1)[0]] for x in res))
		output_text.append(res)
	return output_text

class ModelInterface:
	def __init__(self, path):
		self.model = tf.keras.models.load_model(path, compile=False)
		self.model.compile()

	@staticmethod
	def img2tensor(image: ndarray):
		return tf.expand_dims(image, 0)

class Classifier(ModelInterface):
	def classify(self, image: ndarray):
		tensor_image = self.img2tensor(image)
		preds = self.model(tensor_image)
		return decode_batch_predictions(preds)
