"""
Model Conversion Script for Android Deployment
============================================

This script converts the trained models to TensorFlow Lite format optimized for Android deployment.
It includes INT8 quantization for better performance on mobile devices.

Usage:
    python convert_models.py
    python convert_models.py --detection-model path/to/detection.pt
    python convert_models.py --classification-model path/to/classification.h5
"""

import argparse
import os
import sys
from pathlib import Path
import tensorflow as tf
from ultralytics import YOLO
import numpy as np

def convert_yolo_to_tflite(model_path="Yolov8n_detection/best.pt", output_path="models/"):
    """
    Convert YOLOv8 model to TensorFlow Lite format.
    
    Args:
        model_path (str): Path to the trained YOLOv8 model
        output_path (str): Output directory for converted model
    """
    print("Converting YOLOv8 detection model to TensorFlow Lite...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Load YOLOv8 model
        model = YOLO(model_path)
        
        # Export to TensorFlow Lite with INT8 quantization
        tflite_path = os.path.join(output_path, "fish_detector_int8.tflite")
        
        print("Exporting model...")
        model.export(
            format='tflite',
            imgsz=640,
            int8=True,
            optimize=True,
            dynamic=False,
            simplify=True
        )
        
        # Move the exported file to our output directory
        exported_file = model_path.replace('.pt', '_int8.tflite')
        if os.path.exists(exported_file):
            os.rename(exported_file, tflite_path)
            print(f"✅ Detection model converted successfully: {tflite_path}")
        else:
            print("❌ Failed to find exported model file")
            
    except Exception as e:
        print(f"❌ Error converting detection model: {e}")

def convert_mobilenet_to_tflite(model_path=None, output_path="models/"):
    """
    Convert MobileNetV2 classification model to TensorFlow Lite format.
    
    Args:
        model_path (str): Path to the trained MobileNetV2 model
        output_path (str): Output directory for converted model
    """
    print("Converting MobileNetV2 classification model to TensorFlow Lite...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Find model file if not provided
    if model_path is None:
        model_files = list(Path('mobilenetV3_classfication').glob('*.h5')) + \
                     list(Path('mobilenetV3_classfication').glob('*.keras'))
        if model_files:
            model_path = str(model_files[0])
            print(f"Found model: {model_path}")
        else:
            print("❌ No classification model found. Please train the model first.")
            return
    
    try:
        # Load the trained model
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully: {model_path}")
        
        # Create representative dataset for quantization
        def representative_data_gen():
            for _ in range(100):
                # Generate random data in the expected input shape
                data = np.random.random((1, 224, 224, 3)).astype(np.float32)
                yield [data]
        
        # Convert to TensorFlow Lite with INT8 quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enable INT8 quantization
        converter.target_spec.supported_types = [tf.int8]
        
        # Set representative dataset for quantization
        converter.representative_dataset = representative_data_gen
        
        # Enable GPU acceleration (if available)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        print("Converting model with INT8 quantization...")
        tflite_model = converter.convert()
        
        # Save the model
        tflite_path = os.path.join(output_path, "fish_classifier_int8.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Classification model converted successfully: {tflite_path}")
        
        # Print model info
        model_size = len(tflite_model) / (1024 * 1024)  # Size in MB
        print(f"Model size: {model_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error converting classification model: {e}")

def create_android_assets(output_path="models/"):
    """
    Create Android assets directory with model files.
    
    Args:
        output_path (str): Path to the models directory
    """
    print("Creating Android assets directory...")
    
    # Create Android assets directory structure
    android_assets = "android/app/src/main/assets/"
    os.makedirs(android_assets, exist_ok=True)
    
    # Copy model files to Android assets
    model_files = [
        "fish_detector_int8.tflite",
        "fish_classifier_int8.tflite"
    ]
    
    for model_file in model_files:
        src_path = os.path.join(output_path, model_file)
        dst_path = os.path.join(android_assets, model_file)
        
        if os.path.exists(src_path):
            import shutil
            shutil.copy2(src_path, dst_path)
            print(f"✅ Copied {model_file} to Android assets")
        else:
            print(f"⚠️  Model file not found: {model_file}")

def validate_models(output_path="models/"):
    """
    Validate the converted TensorFlow Lite models.
    
    Args:
        output_path (str): Path to the models directory
    """
    print("Validating converted models...")
    
    # Test detection model
    detection_model_path = os.path.join(output_path, "fish_detector_int8.tflite")
    if os.path.exists(detection_model_path):
        try:
            interpreter = tf.lite.Interpreter(model_path=detection_model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"✅ Detection model validation passed")
            print(f"   Input shape: {input_details[0]['shape']}")
            print(f"   Output shape: {output_details[0]['shape']}")
            
        except Exception as e:
            print(f"❌ Detection model validation failed: {e}")
    else:
        print("⚠️  Detection model not found")
    
    # Test classification model
    classification_model_path = os.path.join(output_path, "fish_classifier_int8.tflite")
    if os.path.exists(classification_model_path):
        try:
            interpreter = tf.lite.Interpreter(model_path=classification_model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"✅ Classification model validation passed")
            print(f"   Input shape: {input_details[0]['shape']}")
            print(f"   Output shape: {output_details[0]['shape']}")
            
        except Exception as e:
            print(f"❌ Classification model validation failed: {e}")
    else:
        print("⚠️  Classification model not found")

def main():
    parser = argparse.ArgumentParser(description="Convert models to TensorFlow Lite for Android")
    parser.add_argument("--detection-model", default="Yolov8n_detection/best.pt", 
                       help="Path to YOLOv8 detection model")
    parser.add_argument("--classification-model", 
                       help="Path to MobileNetV2 classification model")
    parser.add_argument("--output", default="models/", 
                       help="Output directory for converted models")
    parser.add_argument("--create-android-assets", action="store_true",
                       help="Create Android assets directory")
    parser.add_argument("--validate", action="store_true",
                       help="Validate converted models")
    
    args = parser.parse_args()
    
    print("🚀 Starting model conversion for Android deployment...")
    print("=" * 60)
    
    # Convert detection model
    convert_yolo_to_tflite(args.detection_model, args.output)
    print()
    
    # Convert classification model
    convert_mobilenet_to_tflite(args.classification_model, args.output)
    print()
    
    # Create Android assets if requested
    if args.create_android_assets:
        create_android_assets(args.output)
        print()
    
    # Validate models if requested
    if args.validate:
        validate_models(args.output)
        print()
    
    print("✅ Model conversion completed!")
    print(f"Converted models saved to: {args.output}")
    print("\nNext steps:")
    print("1. Copy .tflite files to your Android app's assets folder")
    print("2. Follow the Android integration guide in README.md")
    print("3. Test the models on your target Android device")

if __name__ == "__main__":
    main()
