import tensorflow as tf
from tensorflow import keras
from pathlib import Path

def save(model, path: Path, metadata = {}):
	weights = model.get_weights()
	config = model.get_config()
	config["name"] = {"name": config["name"], **metadata}
	model = tf.keras.models.Model.from_config(config)
	model.set_weights(weights)
	model.save(path)
	return model

def load(path: Path):
	model = keras.models.load_model(path)
	model.meta = model.get_config()["name"]
	model._name = path.name
	return model