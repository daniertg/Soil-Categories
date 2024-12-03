from flask import Flask, request, jsonify
from test_mage import load_model, predict_soil_type

app = Flask(__name__)

# Load model dan label names
MODEL_PATH = r'soil_classifier_rf.pkl'
model, label_names = load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Periksa apakah file gambar disertakan dalam permintaan
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        # Ambil file gambar
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Lakukan prediksi
        predictions = predict_soil_type(model, image_bytes, label_names)
        if predictions is None:
            return jsonify({'error': 'Error during prediction'}), 500
        
        # Format hasil prediksi
        result = [
            {'soil_type': soil_type, 'probability': f"{probability:.2%}"}
            for soil_type, probability in predictions
        ]
        return jsonify({'predictions': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
