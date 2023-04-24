import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def save(model, path: Union[Path, str], metadata={}, frozen=False):
    path = Path(path)
    if frozen:
        metadata["frozen"] = True
        model = keras.models.Model(
            inputs=model.input, outputs=model.output, name=model.name
        )
        model.trainable = False
    weights = model.get_weights()
    config = model.get_config()
    config["name"] = {"name": config["name"], **metadata}
    model = tf.keras.models.Model.from_config(config)
    model.set_weights(weights)
    model.save(path)
    return model


def load(path: Union[Path, str]):
    path = Path(path)
    model = keras.models.load_model(path)
    model.meta = model.get_config()["name"]
    model._name = path.name
    return model


def text2img(text, font_path="./data/fonts/Quicksand.otf", width=128, height=32):
    # Create a new image
    image = Image.new("RGB", (width, height), color="white")

    # Get a drawing context for the image
    draw = ImageDraw.Draw(image)

    # Set up text
    font_size = 1
    font = ImageFont.truetype(font_path, font_size)

    # Resize font to fit text within image
    while font.getsize(text)[0] < width and font.getsize(text)[1] < height:
        font_size += 1
        font = ImageFont.truetype(font_path, font_size)

    # Decrease font size until text fits within image
    while font.getsize(text)[0] > width or font.getsize(text)[1] > height:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)

    # Calculate text position for center alignment
    x = (width - font.getsize(text)[0]) // 2
    y = (height - font.getsize(text)[1]) // 2

    # Draw text onto image
    draw.text((x, y), text, fill="black", font=font)
    return 1 - np.mean(image, axis=2, dtype=float)