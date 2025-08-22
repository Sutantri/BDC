from tensorflow import keras
from tensorflow.keras import layers
import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from yt_dlp import YoutubeDL
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

def get_instagram_direct_url(reel_url):
    ydl_opts = {
        'format': 'mp4',   # pilih format mp4
        'quiet': True,     # output lebih bersih
    }
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(reel_url, download=False)
        video_url = info_dict.get("url", None)   # direct URL video
    return video_url

def load_video_from_url(url, max_frames=MAX_SEQ_LENGTH):
    cap = cv2.VideoCapture(url)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:, :, [2,1,0]]  # BGR ke RGB
        frame = resize_frame(frame)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.array(frames)

# pakai
df = pd.read_csv("datatrain.csv")  # CSV berisi link Reels

for reel_url in df['video_link']:
    direct_url = get_instagram_direct_url(reel_url)   # ambil direct .mp4 URL
    frames = load_video_from_url(direct_url, max_frames=20)