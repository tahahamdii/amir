from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the machine learning model

model = joblib.load('glm_results.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        new_X = data['inputs']
        predictions = model.predict([new_X])
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)
