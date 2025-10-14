import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing import image
# Load model once at startup

model = tf.keras.models.load_model(r"D:\AI_ML_Projects\Brain_Tumor\model\cnn_model.h5")

MODEL_VERSION = '0.11.0'

CLASS_NAMES = labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']


def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    