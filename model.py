import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, roc_auc_score)
import joblib
import torch
import os
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
import json

# Configure paths and ensure directories exist
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the dataset with proper gender handling"""
    # Load data
    data = pd.read_csv("alzheimer_dataset.csv")
    
    # Clean column names and handle gender
    data.columns = data.columns.str.strip()
    
    # Convert gender (M/F) to numeric (1/0)
    if 'M/F' in data.columns:
        data['Gender'] = data['M/F'].str.strip().map({'M': 1, 'F': 0})
    else:
        raise ValueError("Gender column 'M/F' not found in dataset")
    
    # Target processing
    target = 'Group'
    data = data.dropna(subset=[target])
    data[target] = data[target].str.strip().map(
        lambda x: 0 if x.lower() == 'nondemented' else 1
    )
    
    # Select features - match these with your form fields
    features = [
        'Age', 'MMSE', 'Gender', 'EDUC', 'SES', 
        'CDR', 'eTIV', 'nWBV', 'ASF'
    ]
    
    return data, features, target

def handle_class_imbalance(X, y):
    """Apply SMOTE and compute class weights"""
    classes = np.unique(y)
    weights = class_weight.compute_class_weight(
        'balanced', classes=classes, y=y
    )
    class_weights = dict(zip(classes, weights))
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    return X_res, y_res, class_weights

def build_preprocessor(features):
    """Create preprocessing pipeline"""
    numeric_features = [f for f in features if f != 'Gender']
    categorical_features = ['Gender'] if 'Gender' in features else []
    
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

def train_model(X_train, y_train, X_test, y_test, class_weights):
    """Train TabNet model with optimal parameters"""
    tabnet_params = {
        'n_d': 16,
        'n_a': 16,
        'n_steps': 5,
        'gamma': 1.5,
        'lambda_sparse': 1e-4,
        'optimizer_fn': torch.optim.AdamW,
        'optimizer_params': {'lr': 1e-2, 'weight_decay': 1e-4},
        'mask_type': 'entmax',
        'scheduler_params': {'step_size': 5, 'gamma': 0.9},
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'verbose': 10,
    }
    
    model = TabNetClassifier(**tabnet_params)
    
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_name=['train', 'valid'],
        eval_metric=['auc', 'accuracy'],
        max_epochs=200,
        patience=30,
        batch_size=512,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
        weights=class_weights
    )
    
    return model

def evaluate_model(model, X_test, y_test, features):
    """Generate comprehensive evaluation metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": dict(zip(
            [f for f in features if f != 'Group'],
            model.feature_importances_
        ))
    }

def save_artifacts(model, preprocessor, metrics, features):
    """Save all model artifacts"""
    # Save model - use .zip extension directly without creating directory
    model_path = os.path.join(MODEL_DIR, 'tabnet_model')
    model.save_model(model_path)  # This will create tabnet_model.zip automatically
    
    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    
    # Save metrics and configuration
    artifacts = {
        "metrics": metrics,
        "features_used": features,
        "gender_mapping": {"M": 1, "F": 0}
    }
    
    with open(os.path.join(MODEL_DIR, 'model_artifacts.json'), 'w') as f:
        json.dump(artifacts, f, indent=4)

def main():
    # Data loading and preprocessing
    data, features, target = load_and_preprocess_data()
    X = data[features]
    y = data[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing pipeline
    preprocessor = build_preprocessor(features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_train_processed = imputer.fit_transform(X_train_processed)
    X_test_processed = imputer.transform(X_test_processed)
    
    # Handle class imbalance
    X_train_res, y_train_res, class_weights = handle_class_imbalance(
        X_train_processed, y_train
    )
    
    # Model training
    model = train_model(
        X_train_res, y_train_res, 
        X_test_processed, y_test, 
        class_weights
    )
    
    # Evaluation
    metrics = evaluate_model(model, X_test_processed, y_test, features)
    
    # Save artifacts
    save_artifacts(model, preprocessor, metrics, features)
    
    print(f"Model trained successfully. Accuracy: {metrics['accuracy']*100:.2f}%")

if __name__ == '__main__':
    main()