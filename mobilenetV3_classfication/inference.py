"""
MobileNetV2 Fish Classification Inference Script
===============================================

This script provides inference functionality for the trained MobileNetV2 fish classification model.
It can classify fish into 18 different freshness/characterization classes.

Classes:
- Bass GT Sea, Bass Sea
- Black GT Sea Sprat, Black Sea Sprat
- Bream GT Gilt-Head, Bream GT Red Sea, Bream Gilt-Head, Bream Red Sea
- GT Hourse Mackerel, Hourse Mackerel
- GT Mullet Red, GT Mullet Red Striped, Mullet Red, Mullet Red Striped
- GT Shrimp, Shrimp
- GT Trout, Trout

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --batch path/to/folder
    python inference.py --webcam  # for real-time classification
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import sys
from pathlib import Path
import json

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_class_names, print_classification_results

class FishClassifier:
    def __init__(self, model_path=None, input_size=(224, 224), class_names_file="fish_species.txt"):
        """
        Initialize the fish classifier.
        
        Args:
            model_path (str): Path to the trained model (if None, will look for saved model)
            input_size (tuple): Input image size for the model
            class_names_file (str): Path to file containing class names
        """
        self.input_size = input_size
        self.class_names = load_class_names(class_names_file)
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            # Try to find a saved model in the current directory
            model_files = list(Path('.').glob('*.h5')) + list(Path('.').glob('*.keras'))
            if model_files:
                self.model = tf.keras.models.load_model(str(model_files[0]))
                print(f"Loaded model: {model_files[0]}")
            else:
                raise FileNotFoundError("No trained model found. Please train the model first or provide model path.")
        
        print(f"Model loaded successfully. Input shape: {self.model.input_shape}")
        print(f"Number of classes: {len(self.class_names)}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for MobileNetV2 inference.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            numpy array: Preprocessed image
        """
        # Resize image
        if isinstance(image, np.ndarray):
            image = cv2.resize(image, self.input_size)
        else:
            image = image.resize(self.input_size)
        
        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Normalize to [0, 1] and apply MobileNetV2 preprocessing
        image = image.astype(np.float32) / 255.0
        image = preprocess_input(image)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def classify_image(self, image_path, top_k=3, save_result=True, show_result=True):
        """
        Classify fish in a single image.
        
        Args:
            image_path (str): Path to input image
            top_k (int): Number of top predictions to return
            save_result (bool): Whether to save the result
            show_result (bool): Whether to display the result
            
        Returns:
            dict: Classification results with predictions and confidence scores
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        processed_image = self.preprocess_image(image_rgb)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        probabilities = predictions[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_predictions = []
        
        for i, idx in enumerate(top_indices):
            top_predictions.append({
                'rank': i + 1,
                'class_id': int(idx),
                'class_name': self.class_names[idx],
                'confidence': float(probabilities[idx])
            })
        
        # Create result
        result = {
            'image_path': image_path,
            'predictions': top_predictions,
            'best_prediction': top_predictions[0]
        }
        
        # Display results
        print(f"\nClassification Results for: {Path(image_path).name}")
        print_classification_results(top_predictions, top_k)
        
        # Save result if requested
        if save_result:
            output_path = f"classification_result_{Path(image_path).stem}.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {output_path}")
        
        # Show image with prediction if requested
        if show_result:
            self._display_image_with_prediction(image_rgb, top_predictions[0])
        
        return result
    
    def classify_batch(self, folder_path, output_file="batch_results.json"):
        """
        Classify all images in a folder.
        
        Args:
            folder_path (str): Path to folder containing images
            output_file (str): Path to save batch results
            
        Returns:
            list: List of classification results for all images
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
            image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in: {folder_path}")
        
        print(f"Found {len(image_files)} images to classify")
        
        batch_results = []
        for i, image_path in enumerate(image_files):
            print(f"\nProcessing {i+1}/{len(image_files)}: {image_path.name}")
            try:
                result = self.classify_image(str(image_path), save_result=False, show_result=False)
                batch_results.append(result)
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                batch_results.append({
                    'image_path': str(image_path),
                    'error': str(e),
                    'predictions': []
                })
        
        # Save batch results
        with open(output_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"\nBatch classification complete. Results saved to: {output_file}")
        return batch_results
    
    def classify_webcam(self, camera_id=0):
        """
        Real-time fish classification using webcam.
        
        Args:
            camera_id (int): Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        print("Press 'q' to quit webcam classification")
        print("Press 's' to save current frame")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 10th frame to reduce computational load
            if frame_count % 10 == 0:
                try:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Preprocess and predict
                    processed_frame = self.preprocess_image(frame_rgb)
                    predictions = self.model.predict(processed_frame, verbose=0)
                    probabilities = predictions[0]
                    
                    # Get top prediction
                    best_class_id = np.argmax(probabilities)
                    best_confidence = probabilities[best_class_id]
                    best_class_name = self.class_names[best_class_id]
                    
                    # Draw prediction on frame
                    cv2.putText(frame, f"{best_class_name}: {best_confidence:.3f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                except Exception as e:
                    cv2.putText(frame, f"Error: {str(e)[:50]}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow("Fish Classification - Webcam", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"webcam_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _display_image_with_prediction(self, image, prediction):
        """
        Display image with prediction overlay.
        
        Args:
            image: Input image (RGB)
            prediction: Prediction result dictionary
        """
        # Convert RGB to BGR for OpenCV display
        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Add text overlay
        text = f"{prediction['class_name']}: {prediction['confidence']:.3f}"
        cv2.putText(display_image, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display image
        cv2.imshow("Fish Classification", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 Fish Classification Inference")
    parser.add_argument("--model", help="Path to trained model (.h5 or .keras file)")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--batch", help="Path to folder containing images")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time classification")
    parser.add_argument("--output", default="batch_results.json", help="Output file for batch results")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show")
    parser.add_argument("--no-show", action="store_true", help="Don't display results")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    args = parser.parse_args()
    
    try:
        # Initialize classifier
        classifier = FishClassifier(model_path=args.model)
        
        if args.image:
            print(f"Classifying fish in image: {args.image}")
            result = classifier.classify_image(
                args.image,
                top_k=args.top_k,
                save_result=not args.no_save,
                show_result=not args.no_show
            )
            
        elif args.batch:
            print(f"Classifying fish in batch: {args.batch}")
            results = classifier.classify_batch(args.batch, args.output)
            
        elif args.webcam:
            print("Starting webcam classification...")
            classifier.classify_webcam()
            
        else:
            print("Please specify --image, --batch, or --webcam")
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
