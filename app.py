from fastapi import FastAPI, File, UploadFile,Request
from fastapi.responses import JSONResponse
import numpy as np
from model.predict import model, preprocess_image,CLASS_NAMES,MODEL_VERSION
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
        # Read the uploaded file bytes
        image_bytes = await file.read()
        
        if not image_bytes:
            return JSONResponse({"error": "Uploaded file is empty"}, status_code=400)
        
        img_array = preprocess_image(image_bytes)
        prediction = model.predict(img_array)
        result_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "filename": file.filename,
            "prediction": CLASS_NAMES[result_index],
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)