"""
Fish Detection and Classification Pipeline
=========================================

This script combines YOLOv8 fish detection with MobileNetV2 species classification.
It first detects fish using YOLOv8, crops the detected regions, then classifies
each fish using MobileNetV2.

Pipeline:
1. YOLOv8: Detect fish and create bounding boxes
2. Crop: Extract fish regions from bounding boxes
3. MobileNetV2: Classify each cropped fish region

Usage:
    python fish_pipeline.py --image path/to/image.jpg
    python fish_pipeline.py --video path/to/video.mp4
    python fish_pipeline.py --webcam  # for real-time pipeline
"""

import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add subdirectories to path
sys.path.append(str(Path(__file__).parent / "Yolov8n_detection"))
sys.path.append(str(Path(__file__).parent / "mobilenetV3_classfication"))
sys.path.append(str(Path(__file__).parent))

from Yolov8n_detection.inference import FishDetector
from mobilenetV3_classfication.inference import FishClassifier
from utils import print_detection_results, print_classification_results

class FishPipeline:
    def __init__(self, 
                 detection_model="Yolov8n_detection/best.pt",
                 classification_model=None,
                 detection_conf=0.5,
                 classification_conf=0.3):
        """
        Initialize the fish detection and classification pipeline.
        
        Args:
            detection_model (str): Path to YOLOv8 detection model
            classification_model (str): Path to MobileNetV2 classification model
            detection_conf (float): Detection confidence threshold
            classification_conf (float): Classification confidence threshold
        """
        print("Initializing Fish Detection and Classification Pipeline...")
        
        # Initialize detection model
        print("Loading YOLOv8 detection model...")
        self.detector = FishDetector(
            model_path=detection_model,
            conf_threshold=detection_conf
        )
        
        # Initialize classification model
        print("Loading MobileNetV2 classification model...")
        self.classifier = FishClassifier(
            model_path=classification_model,
            class_names_file="mobilenetV3_classfication/fish_species.txt"
        )
        
        self.classification_conf = classification_conf
        print("Pipeline initialized successfully!")
    
    def process_image(self, image_path, save_results=True, show_results=True):
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path (str): Path to input image
            save_results (bool): Whether to save results
            show_results (bool): Whether to display results
            
        Returns:
            dict: Complete pipeline results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\nProcessing image: {Path(image_path).name}")
        print("=" * 60)
        
        # Step 1: Detect fish
        print("Step 1: Detecting fish...")
        detection_result = self.detector.detect_image(
            image_path, 
            save_result=False, 
            show_result=False, 
            crop_fish=True
        )
        
        detections = detection_result['detections']
        cropped_images = detection_result['cropped_images']
        
        print(f"Found {len(detections)} fish")
        
        if not detections:
            print("No fish detected in the image.")
            return {
                'image_path': image_path,
                'detections': [],
                'classifications': [],
                'total_fish': 0
            }
        
        # Step 2: Classify each detected fish
        print("\nStep 2: Classifying detected fish...")
        classifications = []
        
        for i, crop_data in enumerate(cropped_images):
            fish_id = crop_data['fish_id']
            fish_image = crop_data['image']
            bbox = crop_data['bbox']
            detection_conf = crop_data['confidence']
            
            print(f"  Classifying fish {fish_id}...")
            
            # Save temporary cropped image for classification
            temp_path = f"temp_fish_{fish_id}.jpg"
            cv2.imwrite(temp_path, fish_image)
            
            try:
                # Classify the cropped fish
                classification_result = self.classifier.classify_image(
                    temp_path,
                    top_k=3,
                    save_result=False,
                    show_result=False
                )
                
                best_prediction = classification_result['best_prediction']
                
                # Filter by confidence threshold
                if best_prediction['confidence'] >= self.classification_conf:
                    classification = {
                        'fish_id': fish_id,
                        'species': best_prediction['class_name'],
                        'confidence': best_prediction['confidence'],
                        'bbox': bbox,
                        'detection_confidence': detection_conf,
                        'all_predictions': classification_result['predictions']
                    }
                else:
                    classification = {
                        'fish_id': fish_id,
                        'species': 'Unknown',
                        'confidence': best_prediction['confidence'],
                        'bbox': bbox,
                        'detection_confidence': detection_conf,
                        'all_predictions': classification_result['predictions']
                    }
                
                classifications.append(classification)
                
            except Exception as e:
                print(f"    Error classifying fish {fish_id}: {e}")
                classifications.append({
                    'fish_id': fish_id,
                    'species': 'Error',
                    'confidence': 0.0,
                    'bbox': bbox,
                    'detection_confidence': detection_conf,
                    'error': str(e)
                })
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Step 3: Create final results
        results = {
            'image_path': image_path,
            'detections': detections,
            'classifications': classifications,
            'total_fish': len(detections),
            'successful_classifications': len([c for c in classifications if c['species'] != 'Error'])
        }
        
        # Display results
        print(f"\nPipeline Results:")
        print("=" * 60)
        print(f"Total fish detected: {results['total_fish']}")
        print(f"Successfully classified: {results['successful_classifications']}")
        print()
        
        for classification in classifications:
            print(f"Fish {classification['fish_id']}:")
            print(f"  Species: {classification['species']}")
            print(f"  Classification Confidence: {classification['confidence']:.3f}")
            print(f"  Detection Confidence: {classification['detection_confidence']:.3f}")
            bbox = classification['bbox']
            print(f"  Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            print()
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"pipeline_results_{Path(image_path).stem}_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to: {results_file}")
        
        return results
    
    def process_video(self, video_path, output_path=None, show_results=True):
        """
        Process a video through the complete pipeline.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video
            show_results (bool): Whether to display results
            
        Returns:
            list: List of results for each frame
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_results = []
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 10th frame to reduce computational load
            if frame_count % 10 == 0:
                try:
                    # Save temporary frame
                    temp_frame_path = f"temp_frame_{frame_count}.jpg"
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # Process frame through pipeline
                    frame_results = self.process_image(
                        temp_frame_path,
                        save_results=False,
                        show_results=False
                    )
                    
                    all_results.append(frame_results)
                    
                    # Draw results on frame
                    annotated_frame = self._draw_results_on_frame(frame, frame_results['classifications'])
                    
                    # Save frame if output path is provided
                    if output_path:
                        out.write(annotated_frame)
                    
                    # Display frame if requested
                    if show_results:
                        cv2.imshow("Fish Pipeline", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    # Clean up temporary file
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
                
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
            
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        if show_results:
            cv2.destroyAllWindows()
        
        print(f"Video processing complete. Processed {frame_count} frames.")
        return all_results
    
    def _draw_results_on_frame(self, frame, classifications):
        """
        Draw classification results on frame.
        
        Args:
            frame: Input frame
            classifications: List of classification results
            
        Returns:
            numpy array: Annotated frame
        """
        annotated_frame = frame.copy()
        
        for classification in classifications:
            bbox = classification['bbox']
            species = classification['species']
            confidence = classification['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"{species}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame

def main():
    parser = argparse.ArgumentParser(description="Fish Detection and Classification Pipeline")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time pipeline")
    parser.add_argument("--detection-model", default="Yolov8n_detection/best.pt", help="Path to detection model")
    parser.add_argument("--classification-model", help="Path to classification model")
    parser.add_argument("--detection-conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--classification-conf", type=float, default=0.3, help="Classification confidence threshold")
    parser.add_argument("--output", help="Output path for video")
    parser.add_argument("--no-show", action="store_true", help="Don't display results")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = FishPipeline(
            detection_model=args.detection_model,
            classification_model=args.classification_model,
            detection_conf=args.detection_conf,
            classification_conf=args.classification_conf
        )
        
        if args.image:
            results = pipeline.process_image(
                args.image,
                save_results=not args.no_save,
                show_results=not args.no_show
            )
            
        elif args.video:
            results = pipeline.process_video(
                args.video,
                output_path=args.output,
                show_results=not args.no_show
            )
            
        elif args.webcam:
            print("Webcam pipeline not implemented yet. Use individual scripts for now.")
            
        else:
            print("Please specify --image, --video, or --webcam")
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
