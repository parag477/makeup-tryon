document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const webcamContainer = document.getElementById('webcam-container');
    const webcam = document.getElementById('webcam');
    const uploadedImage = document.getElementById('uploaded-image');
    const fileUpload = document.getElementById('file-upload');
    const colorButtons = document.querySelectorAll('.color-btn');
    const makeupToggles = document.querySelectorAll('.switch input[type="checkbox"]');
    const categoryButtons = document.querySelectorAll('.categories button');
    const tryOnButtons = document.querySelectorAll('.try-on-btn');

    let currentImage = null;
    let makeupOptions = {
        lipstick: {
            enabled: true,
            color: [0, 0, 255]
        },
        eyeliner: {
            enabled: true,
            color: [14, 14, 18]
        },
        eyeshadow: {
            enabled: true,
            color: [91, 123, 195]
        },
        blush: {
            enabled: true,
            color: [130, 119, 255]
        }
    };

    // Webcam handling
    async function setupWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcam.srcObject = stream;
            webcam.addEventListener('loadeddata', function() {
                captureAndProcess();
            });
        } catch (err) {
            console.error('Error accessing webcam:', err);
        }
    }

    // Capture and process frame
    async function captureAndProcess() {
        if (webcam.srcObject) {
            const canvas = document.createElement('canvas');
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(webcam, 0, 0);
            
            const imageData = canvas.toDataURL('image/jpeg');
            currentImage = imageData;
            
            await applyMakeup();
            
            if (webcam.srcObject) {
                requestAnimationFrame(captureAndProcess);
            }
        }
    }

    // File upload handling
    fileUpload.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = async function(e) {
                currentImage = e.target.result;
                uploadedImage.src = currentImage;
                webcam.style.display = 'none';
                uploadedImage.style.display = 'block';
                await applyMakeup();
            }
            reader.readAsDataURL(file);
        }
    });

    // Toggle makeup types
    makeupToggles.forEach(toggle => {
        toggle.addEventListener('change', function() {
            const type = this.getAttribute('data-type');
            makeupOptions[type].enabled = this.checked;
            if (currentImage) {
                applyMakeup();
            }
        });
    });

    // Color selection
    colorButtons.forEach(button => {
        button.addEventListener('click', function() {
            const type = this.getAttribute('data-type');
            const rgbValues = JSON.parse(this.getAttribute('data-color'));
            
            // Remove selected class from other buttons of the same type
            document.querySelectorAll(`.color-btn[data-type="${type}"].selected`).forEach(btn => {
                btn.classList.remove('selected');
            });
            
            // Add selected class to clicked button
            this.classList.add('selected');
            
            // Convert RGB to BGR for backend
            const bgrValues = [rgbValues[2], rgbValues[1], rgbValues[0]];
            makeupOptions[type].color = bgrValues;
            
            if (currentImage) {
                applyMakeup();
            }
        });
    });

    // Category selection
    categoryButtons.forEach(button => {
        button.addEventListener('click', function() {
            categoryButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            const category = this.getAttribute('data-category');
            
            // Reset all makeup options to false first
            makeupOptions.lipstick.enabled = false;
            makeupOptions.eyeliner.enabled = false;
            makeupOptions.eyeshadow.enabled = false;
            makeupOptions.blush.enabled = false;
            
            // Enable specific makeup feature based on category
            if (category === 'all') {
                makeupOptions.lipstick.enabled = true;
                makeupOptions.eyeliner.enabled = true;
                makeupOptions.eyeshadow.enabled = true;
                makeupOptions.blush.enabled = true;
            } else if (category === 'lipstick') {
                makeupOptions.lipstick.enabled = true;
            } else if (category === 'eyeliner') {
                makeupOptions.eyeliner.enabled = true;
            } else if (category === 'eyeshadow') {
                makeupOptions.eyeshadow.enabled = true;
            } else if (category === 'blush') {
                makeupOptions.blush.enabled = true;
            }
            
            if (currentImage) {
                applyMakeup();
            }
        });
    });

    // Apply makeup function
    async function applyMakeup() {
        try {
            const response = await fetch('/apply_makeup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: currentImage,
                    lipstick_enabled: makeupOptions.lipstick.enabled,
                    lipstick_color: makeupOptions.lipstick.color,
                    eyeliner_enabled: makeupOptions.eyeliner.enabled,
                    eyeliner_color: makeupOptions.eyeliner.color,
                    eyeshadow_enabled: makeupOptions.eyeshadow.enabled,
                    eyeshadow_color: makeupOptions.eyeshadow.color,
                    blush_enabled: makeupOptions.blush.enabled,
                    blush_color: makeupOptions.blush.color
                })
            });
            
            const data = await response.json();
            if (data.success) {
                uploadedImage.src = data.processed_image;
                uploadedImage.style.display = 'block';
                if (webcam.srcObject) {
                    webcam.style.display = 'none';
                }
            } else {
                console.error('Error applying makeup:', data.error);
            }
        } catch (error) {
            console.error('Error applying makeup:', error);
        }
    }

    // Control button handlers
    document.getElementById('flip-btn').addEventListener('click', async function() {
        const stream = webcam.srcObject;
        if (stream) {
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            
            const videoConstraints = { video: { facingMode: webcam.facingMode === 'user' ? 'environment' : 'user' }};
            try {
                const newStream = await navigator.mediaDevices.getUserMedia(videoConstraints);
                webcam.srcObject = newStream;
                webcam.facingMode = videoConstraints.video.facingMode;
            } catch (err) {
                console.error('Error flipping camera:', err);
            }
        }
    });

    let zoomLevel = 1;
    document.getElementById('zoom-in-btn').addEventListener('click', function() {
        if (zoomLevel < 2) {
            zoomLevel += 0.1;
            webcam.style.transform = `scale(${zoomLevel})`;
            uploadedImage.style.transform = `scale(${zoomLevel})`;
        }
    });

    document.getElementById('zoom-out-btn').addEventListener('click', function() {
        if (zoomLevel > 0.5) {
            zoomLevel -= 0.1;
            webcam.style.transform = `scale(${zoomLevel})`;
            uploadedImage.style.transform = `scale(${zoomLevel})`;
        }
    });

    document.getElementById('capture-btn').addEventListener('click', function() {
        if (webcam.srcObject) {
            const canvas = document.createElement('canvas');
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            canvas.getContext('2d').drawImage(webcam, 0, 0);
            
            currentImage = canvas.toDataURL('image/jpeg');
            applyMakeup();
        }
    });

    // Initialize webcam
    setupWebcam();
});
