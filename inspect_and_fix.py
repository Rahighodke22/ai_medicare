import pandas as pd
import joblib
import pickle

print("="*70)
print("🔍 INSPECTING YOUR DATASET AND MODEL")
print("="*70)

# 1. Check what's in the CSV
print("\n📂 Loading Cleaned_Dataset.csv...")
df = pd.read_csv('Cleaned_Dataset.csv')

print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

print("\n📋 ALL COLUMNS IN DATASET:")
print("-" * 70)
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    unique_count = df[col].nunique()
    print(f"{i:2d}. {col:30s} | Type: {str(dtype):10s} | Unique: {unique_count}")

print("\n" + "="*70)
print("📊 FIRST 5 ROWS OF DATA:")
print("="*70)
print(df.head())

print("\n" + "="*70)
print("📊 DATA TYPES:")
print("="*70)
print(df.dtypes)

# 2. Check what the model expects
print("\n" + "="*70)
print("🤖 CHECKING MODEL REQUIREMENTS:")
print("="*70)

try:
    model = joblib.load('best_model.pkl')
    
    # Try to get feature names from the model
    if hasattr(model, 'feature_names_in_'):
        print("\n✅ Model expects these features (in this exact order):")
        print("-" * 70)
        for i, feat in enumerate(model.feature_names_in_, 1):
            print(f"{i:2d}. {feat}")
        
        print(f"\n📊 Total features expected: {len(model.feature_names_in_)}")
        
        # Save feature names for the app
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(list(model.feature_names_in_), f)
        print("✅ Saved feature names to: feature_names.pkl")
    else:
        print("⚠️  Model doesn't have feature names stored")
        print("   This means it was trained with an older sklearn version")
        
except Exception as e:
    print(f"❌ Could not load model: {e}")

# 3. Check encoder
print("\n" + "="*70)
print("🏷️  CHECKING DISEASE ENCODER:")
print("="*70)

try:
    with open('disease_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    print(f"\n✅ Encoder loaded successfully")
    print(f"📊 Number of diseases: {len(encoder.classes_)}")
    print("\n🏥 Diseases the model can predict:")
    print("-" * 70)
    for i, disease in enumerate(encoder.classes_, 1):
        print(f"{i:2d}. {disease}")
        
except Exception as e:
    print(f"❌ Could not load encoder: {e}")

# 4. Create mapping guide
print("\n" + "="*70)
print("🗺️  CREATING FEATURE MAPPING GUIDE")
print("="*70)

# Common frontend field names
frontend_fields = {
    'fever': 'fever',
    'cough': 'cough', 
    'fatigue': 'fatigue',
    'breathing': 'difficulty_breathing',
    'age': 'age',
    'gender': 'gender',
    'bloodPressure': 'blood_pressure',
    'cholesterol': 'cholesterol_level'
}

print("\nFrontend → Expected Column Mapping:")
print("-" * 70)
for frontend, backend in frontend_fields.items():
    found = backend in df.columns
    status = "✅" if found else "❌"
    print(f"{status} {frontend:20s} → {backend:30s} {'(FOUND)' if found else '(MISSING)'}")

# 5. Identify the target column
print("\n" + "="*70)
print("🎯 IDENTIFYING TARGET COLUMN (Disease/Diagnosis):")
print("="*70)

possible_targets = ['Disease', 'disease', 'diagnosis', 'Diagnosis', 'target', 
                    'outcome', 'label', 'condition', 'illness']

target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        print(f"✅ Found target column: '{col}'")
        print(f"   Unique values: {df[col].nunique()}")
        print(f"   Sample values: {list(df[col].unique()[:5])}")
        break

if not target_col:
    print("⚠️  Could not find obvious target column")
    print("   Possible candidates:")
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() < 50:
            print(f"   - {col} ({df[col].nunique()} unique values)")

print("\n" + "="*70)
print("💡 RECOMMENDATIONS:")
print("="*70)
print("""
1. If model loaded successfully with feature_names.pkl:
   → Update app.py to use exact feature names from the model

2. If feature names don't match:
   → Run the retrain_model.py script to retrain with correct features

3. The frontend is sending these fields:
   fever, cough, fatigue, breathing, age, gender, bloodPressure, cholesterol

4. Make sure your dataset has matching columns (or retrain the model)
""")

print("="*70)