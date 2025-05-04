import tensorflow as tf
import numpy as np
import cv2
import os
import argparse

def test_model(model_path, image_path):
    """
    Test the exported model with a sample image
    
    Args:
        model_path: Path to the exported .h5 model file
        image_path: Path to the test image
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False
    
    try:
        # Load the model
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Process the image
        print(f"Processing image {image_path}...")
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))  # Adjust size based on your model's requirements
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(img)
        
        # Get the class with highest probability
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        # Map class index to label (adjust based on your model's classes)
        class_labels = ['HMPV Negative', 'HMPV Positive']
        result_label = class_labels[class_idx]
        
        # Print results
        print("\nPrediction Results:")
        print(f"Class: {result_label}")
        print(f"Confidence: {confidence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the exported HMPV model with a sample image")
    parser.add_argument("--model", default="models/hmpv_model.h5", help="Path to the exported .h5 model file")
    parser.add_argument("--image", required=True, help="Path to the test image")
    
    args = parser.parse_args()
    
    test_model(args.model, args.image) 