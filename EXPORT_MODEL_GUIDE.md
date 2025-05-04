# Guide to Export Your HMPV Model as an .h5 File

This guide will walk you through the process of exporting your trained HMPV virus analysis model from your Jupyter notebook to an .h5 file that can be used in the web application.

## Step 1: Prepare Your Environment

1. Make sure you have the required packages installed:
   ```bash
   pip install tensorflow numpy pillow opencv-python
   ```

2. Ensure your Jupyter notebook is in the same directory as the `export_model.py` script.

## Step 2: Export the Model from Your Notebook

1. Open your Jupyter notebook (`Correct_Copy_of_covid_19_cnn1 (1).ipynb`).

2. Run all cells up to where your model is fully trained.

3. After training, add a new cell at the end of your notebook with the following code:
   ```python
   # Import the export function
   from export_model import export_model
   
   # Export your model (replace 'model' with your actual model variable name)
   export_model(model, "hmpv_model.h5")
   ```

4. Run this cell to save your model.

5. The model will be saved in a `models` folder in your project directory.

## Step 3: Verify the Exported Model

1. Check that the `models` folder was created and contains your `hmpv_model.h5` file.

2. You can verify the model was exported correctly by running:
   ```python
   import tensorflow as tf
   loaded_model = tf.keras.models.load_model('models/hmpv_model.h5')
   print("Model loaded successfully!")
   ```

## Step 4: Use the Model in the Web Application

1. The web application (`app.py`) is already configured to load the model from the `models/hmpv_model.h5` path.

2. Run the web application:
   ```bash
   python app.py
   ```

3. Access the application at `http://localhost:5000` and test with your images.

## Troubleshooting

If you encounter issues:

1. **Model loading error**: Make sure the model architecture in your notebook matches what's expected by the web application.

2. **Image preprocessing mismatch**: Adjust the `process_image` function in `app.py` to match the preprocessing steps used in your notebook.

3. **Class labels mismatch**: Update the `class_labels` list in `app.py` to match the classes your model was trained on.

## Additional Notes

- If your model uses custom layers or metrics, you may need to register them before loading:
  ```python
  import tensorflow as tf
  from your_custom_layers import CustomLayer
  tf.keras.utils.get_custom_objects()['CustomLayer'] = CustomLayer
  ```

- For large models, consider using TensorFlow's SavedModel format instead of .h5 for better compatibility.

- If your model requires specific preprocessing steps, make sure to document them for future reference. 