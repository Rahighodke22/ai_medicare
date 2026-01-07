import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("="*60)
print("🔄 RETRAINING MEDICARE AI MODEL")
print("="*60)

# Load dataset
print("\n📂 Loading dataset...")
df = pd.read_csv('Cleaned_Dataset.csv')
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Show all columns
print("\n📋 Available columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n📊 First few rows:")
print(df.head())

# Auto-detect disease/target column
disease_col = None
possible_names = ['Disease', 'disease', 'Disease_Name', 'diagnosis', 'Diagnosis', 
                  'target', 'Target', 'label', 'Label', 'outcome', 'Outcome']

for col_name in possible_names:
    if col_name in df.columns:
        disease_col = col_name
        break

# If not found, use the last column as target
if disease_col is None:
    disease_col = df.columns[-1]
    print(f"\n⚠️  Could not auto-detect disease column, using last column: '{disease_col}'")
else:
    print(f"\n✅ Disease column found: '{disease_col}'")

# Separate features and target
y = df[disease_col]
X = df.drop(disease_col, axis=1)

print(f"\n📊 Features shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")
print(f"📊 Unique diseases: {y.nunique()}")
print(f"\n🏥 Diseases in dataset:")
for disease in sorted(y.unique())[:10]:  # Show first 10
    count = (y == disease).sum()
    print(f"  - {disease}: {count} cases")
if y.nunique() > 10:
    print(f"  ... and {y.nunique() - 10} more")

# Handle non-numeric columns
print("\n🔧 Preprocessing data...")

# Check for non-numeric columns
non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()

if non_numeric_cols:
    print(f"⚠️  Found non-numeric columns: {non_numeric_cols}")
    
    for col in non_numeric_cols:
        print(f"   Encoding: {col}")
        # Simple label encoding for categorical variables
        le_temp = LabelEncoder()
        X[col] = le_temp.fit_transform(X[col].astype(str))

# Handle missing values
if X.isnull().any().any():
    print("⚠️  Found missing values, filling with median...")
    X = X.fillna(X.median())

print("✅ Data preprocessed")

# Encode target labels
print("\n🏷️  Encoding disease labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"✅ Encoded {len(label_encoder.classes_)} disease classes")

# Train model
print("\n🤖 Training Random Forest model...")
print("   This may take a few minutes...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

model.fit(X, y_encoded)
print("✅ Model trained successfully!")

# Calculate accuracy
train_accuracy = model.score(X, y_encoded)
print(f"📊 Training accuracy: {train_accuracy*100:.2f}%")

# Save model
print("\n💾 Saving models...")
joblib.dump(model, 'best_model.pkl')
print("✅ Saved: best_model.pkl")

with open('disease_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✅ Saved: disease_encoder.pkl")

# Test the saved model
print("\n🧪 Testing saved model...")
loaded_model = joblib.load('best_model.pkl')
with open('disease_encoder.pkl', 'rb') as f:
    loaded_encoder = pickle.load(f)

# Make a test prediction
test_sample = X.iloc[0:1]
test_pred = loaded_model.predict(test_sample)[0]
test_disease = loaded_encoder.inverse_transform([test_pred])[0]

print(f"✅ Test prediction successful!")
print(f"   Sample disease: {test_disease}")

print("\n" + "="*60)
print("🎉 RETRAINING COMPLETE!")
print("="*60)
print("\n✅ Your models are now compatible with the current sklearn version")
print("✅ You can now run: python app.py")
print("\n💡 The server will now work in FULL MODE with real predictions!")
print("="*60 + "\n")