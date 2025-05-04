import keras
import numpy as np
import os

# This script will help you export your model to an .h5 file
# Follow these steps:

# 1. First, make sure your model is trained and ready to save
# 2. Run this script after training your model in the notebook

def export_model(model, model_name="hmpv_model.h5"):
    """
    Export a trained model to an .h5 file
    
    Args:
        model: The trained Keras model
        model_name: Name of the output .h5 file
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model
    model_path = os.path.join("models", model_name)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Verify the model can be loaded
    try:
        loaded_model = keras.models.load_model(model_path)
        print("Model successfully loaded for verification")
        return True
    except Exception as e:
        print(f"Error verifying model: {e}")
        return False

# Instructions for using this script:
print("""
INSTRUCTIONS FOR EXPORTING YOUR MODEL:

1. Open your Jupyter notebook (Correct_Copy_of_covid_19_cnn1 (1).ipynb)
2. Run all cells up to where your model is trained
3. After training, add a new cell with the following code:

   # Import the export function
   from export_model import export_model
   
   # Export your model (replace 'model' with your actual model variable name)
   export_model(model, "hmpv_model.h5")

4. Run this cell to save your model
5. The model will be saved in a 'models' folder in your project directory
6. The saved model can then be used in the web application

Note: Make sure your model is fully trained before exporting.
""") 