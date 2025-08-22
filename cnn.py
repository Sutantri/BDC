import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import cv2

MAX_SEQ_LENGTH = 20
IMG_SIZE = 128
EPOCHS = 5

train_df = pd.read_csv("datatrain.csv")
test_df = pd.read_csv("datatest.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

resize_layer = layers.Resizing(IMG_SIZE, IMG_SIZE)
def resize_frame(frame):
    resized = resize_layer(frame[None, ...])
    resized = keras.ops.convert_to_numpy(resized)
    resized = keras.ops.squeeze(resized)
    return resized