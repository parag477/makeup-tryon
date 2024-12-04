from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from makeup_app import MakeupApplication
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
makeup_app = MakeupApplication()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image with default makeup
        try:
            processed_image = makeup_app.process_image(filepath)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
            cv2.imwrite(output_path, processed_image)
            
            # Convert the processed image to base64
            _, buffer = cv2.imencode('.jpg', processed_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'filename': filename,
                'processed_image': f'data:image/jpeg;base64,{image_base64}'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/apply_makeup', methods=['POST'])
def apply_makeup():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Prepare makeup options
        makeup_options = {
            'lipstick': {
                'enabled': data.get('lipstick_enabled', True),
                'color': tuple(map(int, data.get('lipstick_color', (0, 0, 255))))
            },
            'eyeliner': {
                'enabled': data.get('eyeliner_enabled', True),
                'color': tuple(map(int, data.get('eyeliner_color', (14, 14, 18))))
            },
            'eyeshadow': {
                'enabled': data.get('eyeshadow_enabled', True),
                'color': tuple(map(int, data.get('eyeshadow_color', (91, 123, 195))))
            },
            'blush': {
                'enabled': data.get('blush_enabled', True),
                'color': tuple(map(int, data.get('blush_color', (130, 119, 255))))
            }
        }

        # Process the image
        processed_image = makeup_app.process_frame(image, makeup_options)
        
        # Convert the processed image back to base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'processed_image': f'data:image/jpeg;base64,{processed_image_base64}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
