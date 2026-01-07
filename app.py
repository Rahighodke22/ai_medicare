# ============================================================================
# FLASK API BACKEND - Connect ML Models to Website
# Fixed version with proper error handling and CORS
# ============================================================================

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import joblib
import warnings
import os
import traceback

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ============================================================================
# LOAD ALL MODELS AND DATA
# ============================================================================

print("Loading models...")
print("➡ CWD:", os.getcwd())
print("➡ Files here:", os.listdir("."))

# Initialize variables
best_model = None
label_encoder = None
medicine_db = None

# ---------- Load best_model.pkl ----------
try:
    best_model = joblib.load('best_model.pkl')
    print("✅ Best model loaded from best_model.pkl")
except Exception as e:
    print("❌ Failed to load best_model.pkl")
    print("   Error:", str(e))
    print("⚠️  Server will run in DEMO MODE")

# ---------- Load disease_encoder.pkl ----------
try:
    with open('disease_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("✅ Label encoder loaded from disease_encoder.pkl")
except Exception as e:
    print("❌ Failed to load disease_encoder.pkl")
    print("   Error:", str(e))

# ---------- Load medicine_database.pkl ----------
try:
    with open('medicine_database.pkl', 'rb') as f:
        medicine_db = pickle.load(f)
    print("✅ Medicine database loaded")
except Exception as e:
    print("⚠️  Creating default medicine database...")
    medicine_db = {}

print("\n🎉 Backend ready!\n")

# ============================================================================
# MEDICINE DATABASE (Comprehensive)
# ============================================================================

COMPLETE_MEDICINE_DB = {
    'Influenza': {
        'medicines': [
            '💊 Oseltamivir (Tamiflu) 75mg - Twice daily for 5 days',
            '💊 Acetaminophen 500mg - Every 6 hours for fever',
            '💊 Ibuprofen 400mg - Every 8 hours for body aches',
            '💧 Plenty of fluids - 8-10 glasses daily'
        ],
        'advice': [
            '🛏️ REST: Get 8-10 hours of sleep',
            '💧 HYDRATION: Drink water, juice, warm liquids',
            '🏠 ISOLATION: Stay home for 7-10 days',
            '🤧 HYGIENE: Cover mouth, wash hands frequently',
            '🌡️ MONITOR: Check temperature twice daily',
            '📞 SEEK HELP: If breathing difficulty or chest pain'
        ]
    },
    'Common Cold': {
        'medicines': [
            '💊 Acetaminophen 500mg - Every 6 hours',
            '💊 Pseudoephedrine 30mg - Every 6 hours',
            '💊 Vitamin C 1000mg - Once daily',
            '🍯 Honey with warm lemon water'
        ],
        'advice': [
            '💧 HYDRATION: Warm liquids',
            '🛏️ REST: Extra sleep',
            '🤧 HYGIENE: Wash hands frequently',
            '🏠 STAY HOME: Avoid spreading',
            '🍯 HONEY: Natural cough suppressant',
            '⏱️ DURATION: Resolves in 7-10 days'
        ]
    },
    'Asthma': {
        'medicines': [
            '💨 Albuterol Inhaler - 2 puffs every 4-6 hours as needed',
            '💨 Fluticasone 110mcg - 2 puffs twice daily',
            '💊 Montelukast 10mg - Once daily at bedtime',
            '🆘 Emergency rescue inhaler always accessible'
        ],
        'advice': [
            '🚭 AVOID: Smoke, dust, pollen, cold air',
            '💨 BREATHING: Practice breathing exercises',
            '🏃 EXERCISE: Moderate with proper warm-up',
            '📊 MONITOR: Keep peak flow readings',
            '🆘 EMERGENCY: Use rescue inhaler, call 911',
            '💊 COMPLIANCE: Never skip medications'
        ]
    },
    'Diabetes': {
        'medicines': [
            '💊 Metformin 500mg - Twice daily with meals',
            '💉 Insulin (if prescribed) - Per schedule',
            '📊 Blood glucose test strips',
            '🍬 Glucose tablets for emergencies'
        ],
        'advice': [
            '🍽️ DIET: Low carb, high fiber meals',
            '🏃 EXERCISE: 30 minutes daily',
            '📊 MONITORING: Test 3-4 times daily',
            '👣 FOOT CARE: Inspect daily',
            '💉 INSULIN: Store properly, rotate sites',
            '🚨 HYPOGLYCEMIA: Treat immediately'
        ]
    },
    'Hypertension': {
        'medicines': [
            '💊 Lisinopril 10mg - Once daily morning',
            '💊 Amlodipine 5mg - Once daily',
            '💊 Losartan 50mg - Once daily',
            '🧂 Low sodium diet (<1500mg/day)'
        ],
        'advice': [
            '🧂 DIET: Drastically limit salt',
            '🏃 EXERCISE: 30-45 minutes daily',
            '📈 MONITOR: Check BP daily',
            '😌 STRESS: Meditation, yoga',
            '⚖️ WEIGHT: Lose 5-10% if overweight',
            '🚫 AVOID: Alcohol, smoking, caffeine'
        ]
    }
}

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except:
        return jsonify({
            'message': '🏥 Medicare AI Backend is running!',
            'status': 'ok',
            'model_loaded': best_model is not None,
            'endpoints': {
                'predict': '/predict (POST)',
                'health': '/health (GET)',
                'models': '/models (GET)'
            }
        })

@app.route('/about')
def about():
    try:
        return render_template('about.html')
    except:
        return jsonify({'message': 'About page not found'})

@app.route('/contact')
def contact():
    try:
        return render_template('contact.html')
    except:
        return jsonify({'message': 'Contact page not found'})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': best_model is not None,
        'encoder_loaded': label_encoder is not None,
        'message': 'Backend is running!'
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'available_models': ['rf', 'gb', 'lr'],
        'current_model': 'best_model',
        'diseases_count': len(label_encoder.classes_) if label_encoder else 0
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Main prediction endpoint"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided',
                'message': 'Please send JSON data'
            }), 400
        
        print("\n" + "="*60)
        print("📥 Received prediction request")
        print("="*60)
        print("Data:", data)
        
        # Extract features with defaults
        fever = int(data.get('fever', 0))
        cough = int(data.get('cough', 0))
        fatigue = int(data.get('fatigue', 0))
        difficulty_breathing = int(data.get('breathing', 0))
        age = int(data.get('age', 30))
        gender = int(data.get('gender', 0))
        blood_pressure = int(data.get('bloodPressure', 1))
        cholesterol = int(data.get('cholesterol', 1))
        model_choice = data.get('model', 'rf')
        
        print(f"\n👤 Patient: Age {age}, Gender {'M' if gender else 'F'}")
        print(f"🔬 Symptoms: Fever={fever}, Cough={cough}, Fatigue={fatigue}, Breathing={difficulty_breathing}")
        
        # DEMO MODE - If model not loaded
        if best_model is None or label_encoder is None:
            print("⚠️  Running in DEMO MODE (model not loaded)")
            
            # Calculate simple risk based on symptoms
            symptom_count = fever + cough + fatigue + difficulty_breathing
            
            if symptom_count >= 3:
                demo_disease = "Influenza"
                confidence = 85.5
                risk = "high"
            elif symptom_count >= 2:
                demo_disease = "Common Cold"
                confidence = 78.3
                risk = "medium"
            else:
                demo_disease = "Common Cold"
                confidence = 65.2
                risk = "low"
            
            treatment = COMPLETE_MEDICINE_DB.get(demo_disease, {
                'medicines': ['💊 Consult doctor for specific treatment'],
                'advice': ['📞 Schedule appointment with healthcare provider']
            })
            
            response = {
                'success': True,
                'disease': demo_disease,
                'confidence': confidence,
                'risk': risk,
                'top5': [
                    {'disease': demo_disease, 'confidence': confidence},
                    {'disease': 'Common Cold', 'confidence': 65.2},
                    {'disease': 'Asthma', 'confidence': 45.1}
                ],
                'medicines': treatment.get('medicines', []),
                'advice': treatment.get('advice', []),
                'model_used': 'demo',
                'note': 'DEMO MODE: Model files need to be retrained with current sklearn version',
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            return jsonify(response), 200
        
        # REAL PREDICTION MODE
        # Get exact training feature order from the model
        try:
            FEATURE_COLUMNS = list(best_model.feature_names_in_)
            print(f"📋 Model expects features: {FEATURE_COLUMNS}")
        except AttributeError:
            # Fallback if model doesn't have feature_names_in_
            print("⚠️ Model doesn't have feature_names_in_, using default order")
            FEATURE_COLUMNS = [
                'fever', 'cough', 'fatigue', 'difficulty_breathing',
                'age_scaled', 'gender', 'bp_scaled', 'chol_scaled'
            ]
        
        # Map frontend input to training features
        input_data = {
            "fever": fever,
            "cough": cough,
            "fatigue": fatigue,
            "difficulty_breathing": difficulty_breathing,
            "age_scaled": age,
            "gender": gender,
            "bp_scaled": blood_pressure,
            "chol_scaled": cholesterol
        }
        
        # Handle any additional features that might be in the model
        # Fill missing features with defaults
        for col in FEATURE_COLUMNS:
            if col not in input_data:
                print(f"⚠️ Adding missing feature '{col}' with default value 0")
                input_data[col] = 0
        
        # Create DataFrame with exact feature order matching training
        input_df = pd.DataFrame(
            [[input_data[col] for col in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS
        )
        
        print(f"📊 Input DataFrame shape: {input_df.shape}")
        print(f"📊 Input DataFrame columns: {list(input_df.columns)}")
        
        # Make prediction
        prediction = best_model.predict(input_df)[0]
        probabilities = best_model.predict_proba(input_df)[0]
        
        # Get disease name
        predicted_disease = label_encoder.classes_[prediction]
        main_confidence = float(probabilities[prediction] * 100)
        
        print(f"✅ Prediction: {predicted_disease} ({main_confidence:.1f}%)")
        
        # Get top 5 predictions
        top_5_idx = np.argsort(probabilities)[-5:][::-1]
        top_5_predictions = []
        
        for idx in top_5_idx:
            disease_name = label_encoder.classes_[idx]
            confidence = float(probabilities[idx] * 100)
            top_5_predictions.append({
                'disease': disease_name,
                'confidence': round(confidence, 2)
            })
        
        # Calculate risk level
        symptom_count = fever + cough + fatigue + difficulty_breathing
        if symptom_count >= 3 and (age > 60 or blood_pressure == 2):
            risk_level = "high"
        elif symptom_count >= 2:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        print(f"⚠️  Risk Level: {risk_level.upper()}")
        
        # Get treatment info
        treatment = COMPLETE_MEDICINE_DB.get(predicted_disease, {
            'medicines': ['💊 Consult doctor for specific treatment'],
            'advice': ['📞 Schedule appointment with healthcare provider']
        })
        
        # Return response
        response = {
            'success': True,
            'disease': predicted_disease,
            'confidence': round(main_confidence, 2),
            'risk': risk_level,
            'top5': top_5_predictions,
            'medicines': treatment.get('medicines', []),
            'advice': treatment.get('advice', []),
            'model_used': model_choice,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        print(f"📤 Sending response")
        print("="*60 + "\n")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Prediction failed. Check server logs for details.'
        }), 500

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 MEDICARE AI - BACKEND SERVER")
    print("="*60)
    print("\n✅ Server starting...")
    print("📡 API available at: http://localhost:5000")
    print("🌐 Frontend should connect to: http://localhost:5000/predict")
    print("\n💡 To test: Send POST request to /predict endpoint")
    
    if best_model is None:
        print("\n⚠️  WARNING: Running in DEMO MODE")
        print("   Reason: Model files could not be loaded")
        print("   Action: Retrain models with current sklearn version")
    
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)