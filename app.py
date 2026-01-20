from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import logging
from logging.handlers import RotatingFileHandler
import os
import json
from urllib.parse import urlencode
import random

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Constants
MODEL_DIR = 'models'
FEATURES = [
    'Age', 'Gender', 'BMI', 'MMSE', 'MemoryComplaints', 'BehavioralProblems',
    'FamilyHistory', 'SleepQuality', 'FunctionalAssessment', 'nWBV', 'CDR',
    'eTIV', 'ASF', 'EDUC', 'SES'
]

# Gender mapping
GENDER_MAPPING = {
    'male': 1,
    'female': 0
}

def load_models():
    try:
        required_files = [
            os.path.join(MODEL_DIR, 'preprocessor.pkl'),
            os.path.join(MODEL_DIR, 'tabnet_model.zip'),
            os.path.join(MODEL_DIR, 'model_artifacts.json')
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file not found: {file_path}")
        
        preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
        model = TabNetClassifier()
        model.load_model(os.path.join(MODEL_DIR, 'tabnet_model.zip'))
        
        with open(os.path.join(MODEL_DIR, 'model_artifacts.json'), 'r') as f:
            artifacts = json.load(f)
        
        return preprocessor, model, artifacts
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        raise

# Load models at startup
try:
    preprocessor, model, model_artifacts = load_models()
    logger.info("Models loaded successfully")
    accuracy = model_artifacts.get('metrics', {}).get('accuracy', 0)
    logger.info(f"Model accuracy: {accuracy:.4f}")
except Exception as e:
    logger.critical(f"Failed to load models: {str(e)}", exc_info=True)
    preprocessor, model, model_artifacts = None, None, {'metrics': {'accuracy': 0}}

@app.route('/')
def index():
    accuracy = model_artifacts.get('metrics', {}).get('accuracy', 0)
    return render_template('index.html', 
                         model_accuracy=f"{accuracy:.2%}",
                         gender_options=GENDER_MAPPING.keys())

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({'error': "Model not loaded properly"}), 500
    
    try:
        # Get data from form or JSON
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        logger.info(f"Received form data: {data}")
        
        # Prepare input data with default None values
        input_data = {feature: None for feature in FEATURES}
        missing_fields = []
        invalid_fields = []
        
        # Process each feature
        for feature in FEATURES:
            value = data.get(feature)
            if value is None or str(value).strip() == '':
                missing_fields.append(feature)
                continue
                
            try:
                if feature == 'Gender':
                    gender_value = str(value).lower()
                    if gender_value in GENDER_MAPPING:
                        input_data[feature] = GENDER_MAPPING[gender_value]
                    else:
                        raise ValueError(f"Invalid gender value: {value}")
                else:
                    input_data[feature] = float(value)
            except (ValueError, KeyError) as e:
                invalid_fields.append(feature)
                logger.warning(f"Invalid value for {feature}: {value} - {str(e)}")
        
        # Check for missing or invalid fields
        if missing_fields or invalid_fields:
            error_msg = ""
            if missing_fields:
                error_msg += f"Missing fields: {', '.join(missing_fields)}. "
            if invalid_fields:
                error_msg += f"Invalid values for: {', '.join(invalid_fields)}."
            logger.warning(error_msg)
            return jsonify({
                'error': "Validation failed",
                'missing_fields': missing_fields,
                'invalid_fields': invalid_fields
            }), 400
        
        # Convert to DataFrame and preprocess
        input_df = pd.DataFrame([input_data])
        
        try:
            processed_data = preprocessor.transform(input_df)
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}", exc_info=True)
            return jsonify({
                'error': "Error processing input data",
                'details': str(e)
            }), 400
        
        # Predict
        try:
            prediction = model.predict(processed_data)
            proba = model.predict_proba(processed_data)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return jsonify({
                'error': "Prediction failed",
                'details': str(e)
            }), 500
        
        # Prepare results
        risk_prob = float(proba[0][1])
        confidence = 'high' if abs(risk_prob - 0.5) > 0.3 else 'medium' if abs(risk_prob - 0.5) > 0.1 else 'low'
        
        # Generate compliment
        compliments = {
            'low': [
                "Great news! Your cognitive health looks strong!",
                "Excellent results! Keep up the healthy habits!",
                "Fantastic! Your brain health is in good shape!",
                "Wonderful! Your results indicate good cognitive function!",
                "Awesome! Your brain health assessment is very positive!"
            ],
            'high': [
                "Thank you for taking this important step for your health!",
                "Knowledge is power - now you can take action!",
                "Early detection is key - you're doing the right thing!",
                "Your awareness is the first step toward better brain health!",
                "This assessment shows your proactive approach to health!"
            ]
        }
        risk_level = 'low' if prediction[0] == 0 else 'high'
        compliment = random.choice(compliments[risk_level])
        
        result = {
            'prediction': 'High Risk' if prediction[0] == 1 else 'Low Risk',
            'probability': f"{risk_prob*100:.1f}%",
            'confidence': confidence,
            'model_accuracy': f"{model_artifacts.get('metrics', {}).get('accuracy', 0)*100:.1f}%",
            'compliment': compliment
        }
        
        logger.info(f"Prediction successful: {result}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'error': "An unexpected error occurred",
            'details': str(e)
        }), 500

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)