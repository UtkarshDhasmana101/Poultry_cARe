from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path
import json

app = Flask(__name__)

# Load assets
model = joblib.load(Path("saved_models/poultry_disease_model.joblib"))
with open(Path("saved_models/model_features.json")) as f:
    features = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        input_data = {f: 0 for f in features}
        input_data.update({k: int(v) for k, v in data.items() if k in features})
        
        proba = model.predict_proba(pd.DataFrame([input_data]))[0]
        max_idx = proba.argmax()
        
        return jsonify({
            "prediction": model.classes_[max_idx],
           
            "probabilities": {
                cls: float(p) for cls, p in zip(model.classes_, proba) 
                if p > 0.05  
            },
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route('/features', methods=['GET'])
def list_features():
    return jsonify({"features": features})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)