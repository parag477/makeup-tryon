// worker.js
self.onmessage = function(event) {
    const imageData = event.data;
    // Perform image processing here
    // For example, apply filters or transformations

    // Send the processed image back
    self.postMessage(imageData);
};