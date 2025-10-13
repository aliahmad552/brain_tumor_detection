import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing import image
# Load model once at startup

model = tf.keras.models.load_model(r"D:\AI_ML_Projects\Brain_Tumor\model\brain_model_full.h5")

MODEL_VERSION = '0.11.0'

CLASS_NAMES = labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array