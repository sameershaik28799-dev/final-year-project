import os
import io
import json
import numpy as np
import h5py
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image

# Force CPU and Legacy Mode
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

MODEL_PATH = 'lung_cancer_model.h5'
IMG_SIZE = (224, 224) 
CLASS_NAMES = ['adenocarcinoma', 'benign', 'large cell carcinoma', 'normal', 'squamous cell carcinoma']

model = None

def load_lung_model():
    global model
    if model is None:
        try:
            import tf_keras as keras
            with h5py.File(MODEL_PATH, 'r') as f:
                model_config = f.attrs.get('model_config')
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                
                config_dict = json.loads(model_config)

                def clean_config(obj):
                    if isinstance(obj, dict):
                        if 'batch_shape' in obj:
                            obj['batch_input_shape'] = obj.pop('batch_shape')
                        if 'dtype' in obj and isinstance(obj['dtype'], dict):
                            obj['dtype'] = 'float32'
                        for key, value in obj.items():
                            clean_config(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            clean_config(item)

                clean_config(config_dict)
                model = keras.models.model_from_json(json.dumps(config_dict))
                model.load_weights(MODEL_PATH)
            print("✅ Model loaded and ready.")
        except Exception as e:
            print(f"❌ Load Error: {e}")
    return model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    net = load_lung_model()
    if net is None: return jsonify({'error': 'Model Error'}), 500

    file = request.files.get('file')
    if not file: return jsonify({'error': 'No file'}), 400

    try:
        # 1. Load and convert to RGB
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize(IMG_SIZE)
        
        # 2. Preprocessing
        # NOTE: If results are still identical, try removing the "/ 255.0" 
        img_array = np.array(img).astype('float32') / 255.0
        
        # 3. Add Batch Dimension (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        # 4. Predict using the call method (often more stable than .predict)
        preds = net(img_batch, training=False).numpy()
        
        # 5. Apply Softmax manually to ensure distribution
        # This prevents the "stuck on one class" issue if the model outputs raw logits
        probabilities = tf.nn.softmax(preds[0]).numpy()
        
        idx = np.argmax(probabilities)
        confidence = float(probabilities[idx])

        # Print to terminal for debugging
        print(f"Probabilities: {probabilities}")
        print(f"Predicted: {CLASS_NAMES[idx]} ({confidence*100:.2f}%)")

        return jsonify({
            'predicted_class': CLASS_NAMES[idx],
            'confidence': f"{confidence * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_lung_model()
    app.run(debug=True, port=5000)