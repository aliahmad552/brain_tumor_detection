# 🧠 Brain Tumor Detection using Deep Learning (FastAPI Web App)

---

## 🌟 Project Overview

Brain tumors are one of the most dangerous health problems in the world. Early detection can save lives, but manually analyzing MRI (Magnetic Resonance Imaging) scans takes a lot of time and experience.  
That’s why this **AI-powered Brain Tumor Detection System** was created — to help doctors, students, and researchers automatically detect brain tumors from MRI images using **Deep Learning**.

This web application allows users to **upload an MRI scan**, and the AI model instantly predicts whether the image shows:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

The goal of this project is to make AI-based medical assistance more **accessible, accurate, and easy to use** through a simple web interface.

---

## 💡 Problem Statement

Detecting brain tumors through MRI images is a **complex and time-consuming task**. Radiologists need years of experience to accurately interpret these scans.  
But even experts can make mistakes because tumor shapes and sizes vary from person to person.

To solve this problem, we used **Artificial Intelligence (AI)** and **Deep Learning** to create a model that can **learn from thousands of MRI images**.  
Now, when a user uploads an MRI image, the AI model can automatically classify it into one of the tumor types with high accuracy.

---

## ⚙️ Techniques & Technologies Used

This project combines **Machine Learning**, **Deep Learning**, and **Web Development** to create a complete end-to-end solution.

### 🧬 Deep Learning Techniques
- **Convolutional Neural Networks (CNNs)** — Used for analyzing medical images. CNNs are excellent at recognizing patterns like shapes, edges, and textures inside brain scans.
- **Data Preprocessing** — Every image is resized to `224x224`, normalized, and converted to RGB format.
- **Softmax Activation Function** — Used in the output layer to calculate the probability of each tumor class.
- **Adam Optimizer & Categorical Crossentropy Loss** — For efficient model training and fast convergence.

### 💻 Web Technologies
- **FastAPI** — A fast and modern Python framework to handle image uploads and API predictions.
- **HTML, CSS, and JavaScript** — Used to create a smooth and modern user interface.
- **Responsive Design** — Works on both computers and mobile phones.
- **File Upload & Prediction System** — Allows drag-and-drop or click-to-upload MRI images.

---

## 🧠 How the Model Works

1. The model takes the uploaded MRI image as input.  
2. It preprocesses the image (resize, normalize, reshape).  
3. The image is passed through the trained CNN model.  
4. The model outputs probabilities for each tumor class.  
5. The class with the highest probability is selected as the **final prediction**.  
6. The confidence score (in %) is also displayed on the UI.

---

## 🌐 How the Web App Works

When you open the app in your browser:
1. You’ll see a clean interface with a drag-and-drop area.  
2. Upload your **MRI scan image**.  
3. Once uploaded, the image preview replaces the drag zone.  
4. Click the **“Predict”** button — the app sends your image to the FastAPI backend.  
5. The backend loads the trained AI model, processes the image, and returns the **prediction result** with a confidence bar.  
6. You can then click **“Predict Again”** to upload another image.

---


## 🚀 How to Run the Project

### 🧰 Step 1: Install Dependencies
Make sure you have **Python 3.10+** installed, then install the required libraries:
```bash
pip install fastapi uvicorn tensorflow pillow numpy python-multipart

```

### 🧠 Step 2: Run the FastAPI Server
``` bash
uvicorn main:app --reload

```
### 🌍 Step 3: Open in Browser

Go to:
```bash
http://127.0.0.1:8000
```
### Step 4: Model Performance
| Metric              | Value                    |
| ------------------- | ------------------------ |
| Training Accuracy   | 98.5%                    |
| Validation Accuracy | 96.2%                    |
| Test Accuracy       | 95.8%                    |
| Model Type          | CNN (TensorFlow / Keras) |
