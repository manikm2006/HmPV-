document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewSection = document.getElementById('previewSection');
    const resultSection = document.getElementById('resultSection');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const predictionResult = document.getElementById('predictionResult');
    const confidenceResult = document.getElementById('confidenceResult');

    // Preview image when selected
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewSection.style.display = 'block';
                resultSection.style.display = 'none';
            }
            reader.readAsDataURL(file);
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        formData.append('file', imageInput.files[0]);

        // Show loading spinner
        loadingSpinner.style.display = 'block';
        resultSection.style.display = 'none';

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                // Update results
                predictionResult.textContent = `Prediction: ${result.prediction}`;
                confidenceResult.textContent = `Confidence: ${(result.confidence * 100).toFixed(2)}%`;
                
                // Show results
                resultSection.style.display = 'block';
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            alert('Error analyzing image. Please try again.');
            console.error('Error:', error);
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });
}); 