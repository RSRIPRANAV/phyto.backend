import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import json


def train():
    print("🚀 Training started...")
    print("Current working directory:", os.getcwd())

    # --- LOAD DATASET ---
    dataset_path = os.path.join(os.getcwd(), 'Dataset.csv')
    print("Looking for dataset at:", dataset_path)

    if not os.path.exists(dataset_path):
        print("❌ Dataset.csv NOT found!")
        return

    print("✅ Dataset.csv found!")

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"❌ Failed to parse CSV: {e}")
        return

    # --- CLEAN COLUMN NAMES ---
    new_cols = []
    for i, col in enumerate(df.columns):
        clean_name = col.split(' ')[0].strip()
        if clean_name in new_cols:
            new_cols.append(f"{clean_name}_{i}")
        else:
            new_cols.append(clean_name)

    df.columns = new_cols
    print(f"Detected columns: {list(df.columns)}")

    # --- FIND REQUIRED COLUMNS ---
    plant_col = next((col for col in df.columns if "Plant" in col), None)
    contam_col = next((col for col in df.columns if "Contaminants" in col), None)

    if plant_col is None or contam_col is None:
        print("❌ Required columns not found (Plant / Contaminants)")
        return

    # --- BUILD PLANT MAP ---
    plant_list = df[[plant_col, contam_col]].dropna().values.tolist()
    plant_map = {}

    for plant, contam in plant_list:
        contam = str(contam).lower()

        if "copper" in contam:
            plant_map.setdefault("Copper", []).append(plant)
        if "cadmium" in contam:
            plant_map.setdefault("Cadmium", []).append(plant)
        if "lead" in contam:
            plant_map.setdefault("Lead", []).append(plant)

    # Save plant mapping
    with open("plant_map.json", "w") as f:
        json.dump(plant_map, f)

    print("🌱 Plant mapping created.")

    # --- SYNTHETIC DATA ---
    print("Generating synthetic training samples...")

    data = []

    for _ in range(5000):
        cu = np.random.uniform(0, 300)
        cd = np.random.uniform(0, 15)
        pb = np.random.uniform(0, 800)

        ratios = {
            'Copper': cu / 50,
            'Cadmium': cd / 1,
            'Lead': pb / 100
        }

        main_contaminant = max(ratios, key=ratios.get)

        data.append([cu, cd, pb, main_contaminant])

    training_df = pd.DataFrame(
        data,
        columns=['Copper', 'Cadmium', 'Lead', 'Contaminant']
    )

    print(f"Generated samples: {len(training_df)}")

    # --- MODEL TRAINING ---
    X = training_df[['Copper', 'Cadmium', 'Lead']]
    y = training_df['Contaminant']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training optimized Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    # --- SAVE MODEL ---
    joblib.dump(model, 'phytorem_rf_model.pkl', compress=3)

    # --- EVALUATION ---
    accuracy = model.score(X_test, y_test)
    print(f"\n✅ Model saved successfully.")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n🎉 Training complete! Model + plant_map.json ready.")


# --- ENTRY POINT ---
if __name__ == "__main__":
    train()