import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import base64
import cv2
import numpy as np
from makeup_app import MakeupApplication
import logging
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize makeup application
makeup_app = None

def get_makeup_app():
    global makeup_app
    if makeup_app is None:
        makeup_app = MakeupApplication()
    return makeup_app

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/apply_makeup', methods=['POST'])
def apply_makeup():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Get the image data
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")

        # Get makeup application instance
        makeup_instance = get_makeup_app()

        # Prepare makeup options
        makeup_options = {
            'lipstick': {
                'enabled': data.get('lipstick_enabled', True),
                'color': tuple(data.get('lipstick_color', [0, 0, 255]))
            },
            'eyeliner': {
                'enabled': data.get('eyeliner_enabled', True),
                'color': tuple(data.get('eyeliner_color', [14, 14, 18]))
            },
            'eyeshadow': {
                'enabled': data.get('eyeshadow_enabled', True),
                'color': tuple(data.get('eyeshadow_color', [91, 123, 195]))
            },
            'blush': {
                'enabled': data.get('blush_enabled', True),
                'color': tuple(data.get('blush_color', [130, 119, 255]))
            }
        }
        
        # Process the image
        processed_image = makeup_instance.process_frame(image, makeup_options)
        
        if processed_image is None:
            raise ValueError("Failed to process image")
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_data = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'processed_image': f'data:image/jpeg;base64,{processed_image_data}'
        })
        
    except Exception as e:
        logger.error(f"Error in apply_makeup: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the app on 0.0.0.0 to make it accessible externally
    app.run(host='0.0.0.0', port=port)