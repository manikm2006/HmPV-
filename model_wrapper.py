import numpy as np
import os
import tensorflow as tf
import traceback
import io
import sys
import random
import cv2

class H5ModelWrapper:
    """
    A wrapper class for loading and using H5 models.
    This class provides a fallback implementation when the model file is not available.
    """
    def __init__(self, model_path):
        """
        Initialize the model wrapper.
        
        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        self.model = None
        
        # Try to load the model using Keras
        try:
            if os.path.exists(model_path):
                print(f"Model file exists at {model_path}, attempting to load...")
                self.model = tf.keras.models.load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
                
                # Capture model summary
                summary_str = io.StringIO()
                sys.stdout = summary_str
                self.model.summary()
                sys.stdout = sys.__stdout__
                print(f"Model summary:\n{summary_str.getvalue()}")
            else:
                print(f"Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Full traceback: {traceback.format_exc()}")
    
    def predict(self, image):
        """
        Make a prediction using the model.
        
        Args:
            image: The input image
            
        Returns:
            Prediction results (HMPV positive or negative)
        """
        # If model is not available, return error
        if self.model is None:
            raise ValueError("Model not loaded. Please ensure the model file exists and is valid.")
        
        try:
            # Ensure image is in the correct format (add batch dimension if needed)
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # Resize to correct dimensions (70x70)
            if image.shape[1:3] != (70, 70):
                resized_images = []
                for i in range(image.shape[0]):
                    img = cv2.resize(image[i], (70, 70))
                    resized_images.append(img)
                image = np.array(resized_images)
            
            # Print image shape for debugging
            print(f"Input image shape: {image.shape}")
            print(f"Input image min: {image.min()}, max: {image.max()}, mean: {image.mean()}")
            
            # Make prediction using the model
            raw_prediction = self.model.predict(image)
            print(f"Raw prediction shape: {raw_prediction.shape}")
            print("Raw prediction values:")
            print(raw_prediction)
            
            # Apply softmax to get probabilities
            probabilities = tf.nn.softmax(raw_prediction[0]).numpy()
            
            # Print class probabilities
            class_names = ['Normal', 'COVID-19', 'Viral Pneumonia', 'HMPV']
            print("\nClass probabilities (after softmax):")
            for i, prob in enumerate(probabilities):
                print(f"{class_names[i]}: {prob:.4f}")
            
            # Get the HMPV probability (4th class, index 3)
            hmpv_probability = probabilities[3]
            
            # Create binary prediction based on HMPV probability
            # Using a lower threshold of 0.2 to be more sensitive to HMPV cases
            if hmpv_probability > 0.2:
                # HMPV Positive
                binary_prediction = np.array([[0.0, 1.0]])
                print(f"HMPV Positive with probability: {hmpv_probability:.4f}")
            else:
                # HMPV Negative
                binary_prediction = np.array([[1.0, 0.0]])
                print(f"HMPV Negative with probability: {1-hmpv_probability:.4f}")
                
            return binary_prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Full traceback: {traceback.format_exc()}")
            raise ValueError("Error making prediction with the model. Please try again.") 