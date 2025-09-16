from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("../models/cifar10_cnn")

app = FastAPI()

# Define input schema
class ImageData(BaseModel):
    data: list  # expects a 32x32x3 image as nested lists

@app.post("/predict")
def predict(image: ImageData):
    x = np.array(image.data, dtype=np.float32)
    x = x.reshape((1, 32, 32, 3))  # batch dimension
    x = x / 255.0  # normalize if needed
    preds = model.predict(x)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return {"class_id": class_id, "confidence": confidence}
