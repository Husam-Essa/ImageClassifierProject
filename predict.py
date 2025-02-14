import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Utility function to load and preprocess the image
def process_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Utility function to load the JSON file for category names
def load_category_names(category_names_path):
    with open(category_names_path, 'r') as f:
        category_names = json.load(f)
    return category_names

# Predict the top K classes
def predict_top_k(model, processed_image, top_k):
    predictions = model.predict(processed_image)
    top_k_indices = predictions[0].argsort()[-top_k:][::-1]
    top_k_probs = predictions[0][top_k_indices]
    return top_k_indices, top_k_probs

# Main function for prediction
def predict(image_path, model_path, top_k=1, category_names_path=None):
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the image
    processed_image = process_image(image_path)
    
    # Make prediction
    top_k_indices, top_k_probs = predict_top_k(model, processed_image, top_k)
    
    # Load category names if provided
    category_names = None
    if category_names_path:
        category_names = load_category_names(category_names_path)

    # Print results
    for i in range(top_k):
        class_index = top_k_indices[i]
        probability = top_k_probs[i]
        class_name = category_names[str(class_index)] if category_names else str(class_index)
        print(f"{class_name}: {probability:.4f}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Predict flower classes from an image")
    parser.add_argument('image_path', type=str, help="Path to the image file")
    parser.add_argument('model_path', type=str, help="Path to the saved Keras model")
    parser.add_argument('--top_k', type=int, default=1, help="Return the top K most likely classes")
    parser.add_argument('--category_names', type=str, help="Path to a JSON file mapping labels to flower names")

    # Parse arguments
    args = parser.parse_args()
    
    # Call the predict function
    predict(args.image_path, args.model_path, top_k=args.top_k, category_names_path=args.category_names)
