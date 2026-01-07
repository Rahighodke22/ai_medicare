# 🏥 MEDICARE AI – Disease Prediction & Recommendation System

MEDICARE AI is a **machine learning–powered healthcare backend** built with **Flask** that predicts possible diseases based on patient symptoms and demographics, assesses **risk level**, and provides **medicine recommendations and medical advice**.

This system is designed for **educational and prototype purposes** and demonstrates how ML models can assist in preliminary medical decision support.

---

## 🚀 Features

- 🧠 **ML-based Disease Prediction**
- 📊 **Confidence Scores & Top-5 Predictions**
- ⚠️ **Risk Level Assessment (Low / Medium / High)**
- 💊 **Medicine & Treatment Recommendations**
- 🏥 **Health Monitoring API**
- 🌐 **CORS-enabled REST API**
- 📈 **Model & Encoder Health Checks**

---

## 🏗️ System Architecture

Client (Frontend / Postman)
↓
Flask REST API
↓
Input Validation & Preprocessing
↓
Machine Learning Model (best_model.pkl)
↓
Disease Classification
↓
Risk Assessment Logic
↓
Medicine & Advice Mapping
↓
JSON Response


---

## 🧰 Technology Stack

| Layer | Technology |
|----|----|
| Backend | Flask |
| ML Model | scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Loading | Joblib, Pickle |
| API Security | Flask-CORS |
| Language | Python 3.11 |

---

## 📂 Project Structure



medicare/

│
├── app.py

├── best_model.pkl

├── disease_encoder.pkl

├── medicine_database.pkl (optional)

├── requirements.txt

├── templates/

│ ├── index.html

│ ├── about.html

│ └── contact.html

├── venv/

└── README.md



---

## 💻 Installation & Setup

### ✅ Prerequisites
- Python **3.11**
- `pip`
- Virtual Environment (recommended)

---

### 🔹 Step 1: Clone / Open Project
```bash
cd medicare

🔹 Step 2: Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate

🔹 Step 3: Install Dependencies
pip install -r requirements.txt

🔹 Step 4: Run Server
python app.py


then click on the
🌐 Frontend should connect to: http://localhost:5000/predict



Server will start at:

http://localhost:5000
