import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from yt_dlp import YoutubeDL
import numpy as np
import imageio
import cv2
import os
from sklearn.preprocessing import LabelEncoder

# Constants
MAX_SEQ_LENGTH = 20
IMG_SIZE = 128
NUM_FEATURES = 2048  # Feature vector size from feature extractor
EPOCHS = 5

# Load data
try:
    train_df = pd.read_csv("D:/BDC/datatrain.csv")
    test_df = pd.read_csv("D:/BDC/datatest.csv")
    print(f"Total videos for training: {len(train_df)}")
    print(f"Total videos for testing: {len(test_df)}")
except FileNotFoundError:
    print("CSV files not found. Please ensure datatrain.csv and datatest.csv exist.")

# Create label processor
def create_label_processor(train_df):
    """Create label encoder for video labels"""
    label_encoder = LabelEncoder()
    if 'label' in train_df.columns:
        label_encoder.fit(train_df['label'])
    else:
        # If no label column, create dummy labels
        unique_labels = ['action1', 'action2', 'action3']  # Replace with actual labels
        label_encoder.fit(unique_labels)
    return label_encoder

# Initialize label processor
if 'train_df' in locals():
    label_processor = create_label_processor(train_df)
else:
    # Create dummy label processor if CSV not found
    from sklearn.preprocessing import LabelEncoder
    label_processor = LabelEncoder()
    label_processor.fit(['action1', 'action2', 'action3'])

# Create feature extractor using pre-trained model
def build_feature_extractor():
    """Build feature extractor using pre-trained CNN"""
    base_model = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False
    
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = keras.applications.inception_v3.preprocess_input(inputs)
    outputs = base_model(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# Frame processing utilities
resize_layer = layers.Resizing(IMG_SIZE, IMG_SIZE)

def resize_frame(frame):
    """Resize frame to target size"""
    frame = tf.cast(frame, tf.float32)
    resized = resize_layer(frame[None, ...])
    resized = tf.squeeze(resized, axis=0)
    return resized.numpy()

def get_instagram_direct_url(reel_url):
    """Get direct video URL from Instagram reel"""
    ydl_opts = {
        'format': 'mp4',
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(reel_url, download=False)
            video_url = info_dict.get("url", None)
        return video_url
    except Exception as e:
        print(f"Error extracting URL: {e}")
        return None

def load_video_from_url(url, max_frames=MAX_SEQ_LENGTH):
    """Load video frames from URL"""
    if url is None:
        return np.array([])
    
    cap = cv2.VideoCapture(url)
    frames = []
    
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_frame(frame)
            frames.append(frame)
    finally:
        cap.release()
    
    return np.array(frames)

def load_video(path, offload_to_cpu=True):
    """Load video from file path"""
    cap = cv2.VideoCapture(path)
    frames = []
    
    try:
        while len(frames) < MAX_SEQ_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_frame(frame)
            frames.append(frame)
    finally:
        cap.release()
    
    return np.array(frames)

# Positional Embedding Layer
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, self.compute_dtype)
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config

# Transformer Encoder Layer
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config

# Model building function
def get_compiled_model(shape):
    """Build and compile the transformer model"""
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 4
    num_heads = 8
    classes = len(label_processor.classes_)

    inputs = keras.Input(shape=shape)
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Data preparation functions
def prepare_all_videos(df, is_training=True):
    """Prepare all videos for training/testing"""
    num_samples = len(df)
    video_data = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    labels = []

    for idx, row in df.iterrows():
        if 'video_link' in row:
            # For URL-based videos
            direct_url = get_instagram_direct_url(row['video_link'])
            frames = load_video_from_url(direct_url)
        else:
            # For file-based videos
            video_path = row['video_name'] if 'video_name' in row else row['video_path']
            frames = load_video(video_path)
        
        if len(frames) > 0:
            frame_features = prepare_single_video(frames)
            video_data[idx] = frame_features[0]
        
        # Get label
        if 'label' in row:
            labels.append(label_processor.transform([row['label']])[0])
        else:
            labels.append(0)  # Default label if not available
    
    return video_data, np.array(labels)

def prepare_single_video(frames):
    """Prepare a single video for prediction"""
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Pad shorter videos
    if len(frames) < MAX_SEQ_LENGTH:
        diff = MAX_SEQ_LENGTH - len(frames)
        padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        frames = np.concatenate([frames, padding])

    frames = frames[None, ...]

    # Extract features from frames
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            if np.mean(batch[j, :]) > 0.0:
                frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :], verbose=0
                )
            else:
                frame_features[i, j, :] = 0.0

    return frame_features

# Training function
def run_experiment():
    """Run the training experiment"""
    # Try to load pre-saved data first
    try:
        train_data = np.load("train_data.npy")
        train_labels = np.load("train_labels.npy")
        test_data = np.load("test_data.npy")
        test_labels = np.load("test_labels.npy")
        print("Loaded pre-processed data from .npy files")
    except FileNotFoundError:
        print("Pre-processed data not found. Processing videos...")
        if 'train_df' in locals() and 'test_df' in locals():
            train_data, train_labels = prepare_all_videos(train_df, is_training=True)
            test_data, test_labels = prepare_all_videos(test_df, is_training=False)
            
            # Save processed data
            np.save("train_data.npy", train_data)
            np.save("train_labels.npy", train_labels)
            np.save("test_data.npy", test_data)
            np.save("test_labels.npy", test_labels)
        else:
            print("No training data available")
            return None

    print(f"Frame features in train set: {train_data.shape}")
    print(f"Frame features in test set: {test_data.shape}")

    filepath = "video_classifier_weights.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    model = get_compiled_model(train_data.shape[1:])
    
    history = model.fit(
        train_data,
        train_labels,
        validation_split=0.15,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        verbose=1
    )

    model.load_weights(filepath)
    _, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model

# Prediction function
def predict_action(video_path):
    """Predict action for a single video"""
    class_vocab = label_processor.classes_

    if os.path.exists(video_path):
        frames = load_video(video_path, offload_to_cpu=True)
    else:
        print(f"Video file not found: {video_path}")
        return None
    
    if len(frames) == 0:
        print("No frames extracted from video")
        return None

    frame_features = prepare_single_video(frames)
    probabilities = trained_model.predict(frame_features, verbose=0)[0]

    print(f"\nPrediction results for {video_path}:")
    plot_x_axis, plot_y_axis = [], []

    for i in np.argsort(probabilities)[::-1]:
        plot_x_axis.append(class_vocab[i])
        plot_y_axis.append(probabilities[i])
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(plot_x_axis, plot_y_axis)
    plt.xlabel("Class Label")
    plt.ylabel("Probability")
    plt.title("Action Classification Results")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return frames

# Visualization utility
def to_gif(images, filename="animation.gif"):
    """Convert image sequence to GIF"""
    if len(images) == 0:
        print("No images to convert")
        return
    
    converted_images = images.astype(np.uint8)
    imageio.mimsave(filename, converted_images, fps=10)
    print(f"GIF saved as {filename}")

# Main execution
if __name__ == "__main__":
    print("Starting video classification training...")
    
    # Train the model
    trained_model = run_experiment()
    
    if trained_model is not None:
        print("\nModel training completed!")
        
        # Test prediction if test data is available
        if 'test_df' in locals() and len(test_df) > 0:
            test_video = np.random.choice(test_df["video_name"].values.tolist())
            print(f"\nTesting with video: {test_video}")
            
            if os.path.exists(test_video):
                test_frames = predict_action(test_video)
                if test_frames is not None:
                    to_gif(test_frames[:MAX_SEQ_LENGTH])
            else:
                print(f"Test video not found: {test_video}")
    else:
        print("Model training failed - no data available")