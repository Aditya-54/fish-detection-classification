"""
Shared utility functions for Fish Detection and Classification Project
==================================================================

This module contains common utility functions used across both detection
and classification models.
"""

import os
from pathlib import Path

def load_class_names(file_path="fish_species.txt"):
    """
    Load class names from a text file.
    
    The file format should be:
    1|Class Name 1
    2|Class Name 2
    ...
    
    Or simply:
    Class Name 1
    Class Name 2
    ...
    
    Args:
        file_path (str): Path to the class names file
        
    Returns:
        list: List of class names
    """
    class_names = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    # Extract class name (remove numbering if present)
                    if '|' in line:
                        class_name = line.split('|', 1)[1].strip()
                    else:
                        class_name = line.strip()
                    if class_name:
                        class_names.append(class_name)
    except FileNotFoundError:
        print(f"Warning: Class names file '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error loading class names: {e}")
        return []
    
    return class_names

def get_model_info(model_path):
    """
    Get basic information about a model file.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        dict: Model information including file size, type, etc.
    """
    if not os.path.exists(model_path):
        return {"error": "Model file not found"}
    
    file_path = Path(model_path)
    file_size = file_path.stat().st_size
    
    info = {
        "file_name": file_path.name,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "file_extension": file_path.suffix,
        "exists": True
    }
    
    return info

def create_output_directory(base_name="output"):
    """
    Create a unique output directory with timestamp.
    
    Args:
        base_name (str): Base name for the output directory
        
    Returns:
        str: Path to the created directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def validate_image_file(file_path):
    """
    Validate if a file is a valid image file.
    
    Args:
        file_path (str): Path to the file to validate
        
    Returns:
        bool: True if valid image file, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    file_extension = Path(file_path).suffix.lower()
    
    return file_extension in valid_extensions

def get_supported_image_files(directory):
    """
    Get all supported image files in a directory.
    
    Args:
        directory (str): Path to directory to search
        
    Returns:
        list: List of valid image file paths
    """
    if not os.path.exists(directory):
        return []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for ext in valid_extensions:
        image_files.extend(Path(directory).glob(f"*{ext}"))
        image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
    
    return [str(f) for f in image_files]

def format_confidence(confidence, decimal_places=4):
    """
    Format confidence score for display.
    
    Args:
        confidence (float): Confidence score
        decimal_places (int): Number of decimal places
        
    Returns:
        str: Formatted confidence string
    """
    return f"{confidence:.{decimal_places}f} ({confidence*100:.2f}%)"

def print_classification_results(results, top_k=3):
    """
    Print classification results in a formatted way.
    
    Args:
        results (list): List of prediction results
        top_k (int): Number of top results to display
    """
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results[:top_k]):
        rank = result.get('rank', i+1)
        class_name = result.get('class_name', 'Unknown')
        confidence = result.get('confidence', 0.0)
        
        print(f"{rank:2d}. {class_name:<30} {format_confidence(confidence)}")
    
    if len(results) > top_k:
        print(f"... and {len(results) - top_k} more results")
    
    print("="*60)

def print_detection_results(detections):
    """
    Print detection results in a formatted way.
    
    Args:
        detections (list): List of detection results
    """
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    
    if not detections:
        print("No fish detected in the image.")
        return
    
    for i, det in enumerate(detections):
        class_name = det.get('class_name', 'Unknown')
        confidence = det.get('confidence', 0.0)
        bbox = det.get('bbox', [0, 0, 0, 0])
        
        print(f"{i+1:2d}. {class_name:<20} {format_confidence(confidence)}")
        print(f"    Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    
    print("="*60)
