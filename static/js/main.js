document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const webcamContainer = document.getElementById('webcam-container');
    const webcam = document.getElementById('webcam');
    const uploadedImage = document.getElementById('uploaded-image');
    const fileUpload = document.getElementById('file-upload');
    const categoryButtons = document.querySelectorAll('.categories button');
    const colorButtons = document.querySelectorAll('.color-btn');
    const tryOnButtons = document.querySelectorAll('.try-on-btn');

    let currentImage = null;
    let makeupOptions = {
        lipstick_enabled: true,
        lipstick_color: [0, 0, 255],
        eyeliner_enabled: true,
        eyeliner_color: [14, 14, 18],
        eyeshadow_enabled: true,
        eyeshadow_color: [91, 123, 195],
        blush_enabled: true,
        blush_color: [130, 119, 255]
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
            
            try {
                const response = await fetch('/apply_makeup', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: imageData,
                        ...makeupOptions
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    uploadedImage.src = data.processed_image;
                    uploadedImage.style.display = 'block';
                    webcam.style.display = 'none';
                }
            } catch (error) {
                console.error('Error processing frame:', error);
            }
            
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
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    if (data.success) {
                        uploadedImage.src = data.processed_image;
                    }
                } catch (error) {
                    console.error('Error uploading file:', error);
                }
            }
            reader.readAsDataURL(file);
        }
    });

    // Category selection
    categoryButtons.forEach(button => {
        button.addEventListener('click', function() {
            categoryButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            const category = this.getAttribute('data-category');
            
            // Reset all makeup options to false first
            makeupOptions.lipstick_enabled = false;
            makeupOptions.eyeliner_enabled = false;
            makeupOptions.eyeshadow_enabled = false;
            makeupOptions.blush_enabled = false;
            
            // Enable specific makeup feature based on category
            if (category === 'all') {
                makeupOptions.lipstick_enabled = true;
                makeupOptions.eyeliner_enabled = true;
                makeupOptions.eyeshadow_enabled = true;
                makeupOptions.blush_enabled = true;
            } else if (category === 'lipstick') {
                makeupOptions.lipstick_enabled = true;
            } else if (category === 'eyeliner') {
                makeupOptions.eyeliner_enabled = true;
            } else if (category === 'eyeshadow') {
                makeupOptions.eyeshadow_enabled = true;
            } else if (category === 'blush') {
                makeupOptions.blush_enabled = true;
            }
            
            if (currentImage) {
                applyMakeup();
            }
        });
    });

    // Color selection
    colorButtons.forEach(button => {
        button.addEventListener('click', function() {
            colorButtons.forEach(btn => btn.classList.remove('selected'));
            this.classList.add('selected');
            
            const activeCategory = document.querySelector('.categories button.active').getAttribute('data-category');
            const rgbValues = JSON.parse(this.getAttribute('data-color'));
            
            // Convert RGB to BGR (swap first and last values)
            const bgrValues = [rgbValues[2], rgbValues[1], rgbValues[0]];
            
            // Update makeup options based on category
            if (activeCategory === 'all' || activeCategory === 'lipstick') {
                makeupOptions.lipstick_color = bgrValues;
            }
            if (activeCategory === 'all' || activeCategory === 'eyeliner') {
                makeupOptions.eyeliner_color = bgrValues;
            }
            if (activeCategory === 'all' || activeCategory === 'eyeshadow') {
                makeupOptions.eyeshadow_color = bgrValues;
            }
            if (activeCategory === 'all' || activeCategory === 'blush') {
                makeupOptions.blush_color = bgrValues;
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
                    ...makeupOptions
                })
            });
            
            const data = await response.json();
            if (data.success) {
                uploadedImage.src = data.processed_image;
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
