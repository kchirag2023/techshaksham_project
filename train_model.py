import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Load dataset
df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")  # Update with actual file name

# Extract features and target
X = df.iloc[:, :-1]  # Symptoms
y = df.iloc[:, -1]   # Diseases

# Encode disease labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert categorical symptom data (if any) into numerical format
X = X.applymap(lambda x: 1 if x == "yes" else 0)  # Convert 'yes'/'no' to 1/0

# Handle class imbalance
class_weights = compute_class_weight("balanced", classes=np.unique(y_encoded), y=y_encoded)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Train Random Forest model with better parameters
model = RandomForestClassifier(
    n_estimators=200,      # Increase trees
    max_depth=None,        # Let trees grow fully
    class_weight=class_weights_dict,  # Handle imbalance
    random_state=42
)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Random Forest model trained and saved successfully!")
