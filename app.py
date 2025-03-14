from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load trained model safely
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict.get('model', None)
    if model is None:
        raise ValueError("Model not found in the loaded file")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Labels dictionary
labels_dict = {
    0: 'ક', 1: 'ખ', 2: 'ગ', 3: 'ઘ', 4: 'ચ', 5: 'છ', 6: 'જ', 7: 'ઝ', 8: 'ટ', 9: 'ઠ',
    10: 'ડ', 11: 'ઢ', 12: 'ણ', 13: 'ત', 14: 'થ', 15: 'દ', 16: 'ધ', 17: 'ન', 18: 'પ',
    19: 'ફ', 20: 'બ', 21: 'ભ', 22: 'મ', 23: 'ય', 24: 'ર', 25: 'લ', 26: 'વ', 27: 'શ',
    28: 'ષ', 29: 'સ', 30: 'હ', 31: 'ળ', 32: 'ક્ષ', 33: 'જ્ઞ'
}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Read and process image
        file = request.files.get('image')
        if file is None:
            return jsonify({'error': 'No image provided'}), 400

        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Convert to RGB and process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_, y_ = [], []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            if len(data_aux) == 42:  # Ensure correct feature size
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                return jsonify({'prediction': predicted_character})

        return jsonify({'prediction': 'No hand detected'})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
