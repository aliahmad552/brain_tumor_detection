from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from model.predict import model, preprocess_image,CLASS_NAMES,MODEL_VERSION
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Brain Tumor Detection API is running 🚀"}

@app.get("/health")
def health_check():
    return {
        'status':'OK',
        'version':MODEL_VERSION,
        'model_loaded':model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        # Predict
        prediction = model.predict(img_array)
        result_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        return JSONResponse({
            "filename": file.filename,
            "prediction": CLASS_NAMES[result_index],
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

