"""
YOLOv8 Fish Detection Inference Script
=====================================

This script provides inference functionality for the trained YOLOv8 fish detection model.
It detects fish and creates bounding boxes for further classification.

The YOLOv8 model is trained to detect fish as a single class "fish" and provides
bounding boxes that can be used for species classification with MobileNetV2.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --video path/to/video.mp4
    python inference.py --webcam  # for real-time webcam detection
    python inference.py --image path/to/image.jpg --crop  # crop detected fish regions
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_class_names, print_detection_results

class FishDetector:
    def __init__(self, model_path="best.pt", conf_threshold=0.5):
        """
        Initialize the fish detector with trained model.
        
        Args:
            model_path (str): Path to the trained YOLOv8 model
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # YOLOv8 is trained to detect fish as a single class
        self.class_name = "fish"
        
    def detect_image(self, image_path, save_result=True, show_result=True, crop_fish=False):
        """
        Detect fish in a single image and optionally crop detected regions.
        
        Args:
            image_path (str): Path to input image
            save_result (bool): Whether to save the result image
            show_result (bool): Whether to display the result
            crop_fish (bool): Whether to crop detected fish regions
            
        Returns:
            dict: Detection results with bounding boxes and cropped images
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run inference
        results = self.model(image_path, conf=self.conf_threshold)
        
        # Process results
        detections = []
        cropped_images = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Ensure coordinates are within image bounds
                    h, w = original_img.shape[:2]
                    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_name': self.class_name,
                        'fish_id': i + 1
                    }
                    detections.append(detection)
                    
                    # Crop fish region if requested
                    if crop_fish:
                        cropped_fish = original_img[y1:y2, x1:x2]
                        if cropped_fish.size > 0:  # Ensure crop is valid
                            cropped_images.append({
                                'fish_id': i + 1,
                                'image': cropped_fish,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(confidence)
                            })
        
        # Save and display results
        if save_result or show_result:
            annotated_img = results[0].plot()
            
            if save_result:
                output_path = f"detection_result_{Path(image_path).stem}.jpg"
                cv2.imwrite(output_path, annotated_img)
                print(f"Detection result saved to: {output_path}")
                
            if show_result:
                cv2.imshow("Fish Detection", annotated_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        # Save cropped fish images if requested
        if crop_fish and cropped_images:
            crop_dir = f"cropped_fish_{Path(image_path).stem}"
            os.makedirs(crop_dir, exist_ok=True)
            
            for crop in cropped_images:
                crop_filename = f"fish_{crop['fish_id']}_conf_{crop['confidence']:.2f}.jpg"
                crop_path = os.path.join(crop_dir, crop_filename)
                cv2.imwrite(crop_path, crop['image'])
                print(f"Cropped fish saved to: {crop_path}")
        
        return {
            'detections': detections,
            'cropped_images': cropped_images if crop_fish else [],
            'total_fish': len(detections)
        }
    
    def detect_video(self, video_path, output_path=None, show_result=True):
        """
        Detect fish in a video file.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video (optional)
            show_result (bool): Whether to display the result
            
        Returns:
            list: List of detection results for each frame
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
        
        all_detections = []
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference on frame
            results = self.model(frame, conf=self.conf_threshold)
            
            # Process detections for this frame
            frame_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        frame_detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.class_names[class_id]
                        })
            
            all_detections.append(frame_detections)
            
            # Draw detections on frame
            annotated_frame = results[0].plot()
            
            # Save frame if output path is provided
            if output_path:
                out.write(annotated_frame)
            
            # Display frame if requested
            if show_result:
                cv2.imshow("Fish Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        if show_result:
            cv2.destroyAllWindows()
        
        print(f"Video processing complete. Processed {frame_count} frames.")
        return all_detections
    
    def detect_webcam(self, camera_id=0):
        """
        Real-time fish detection using webcam.
        
        Args:
            camera_id (int): Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        print("Press 'q' to quit webcam detection")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model(frame, conf=self.conf_threshold)
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Display frame
            cv2.imshow("Fish Detection - Webcam", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Fish Detection Inference")
    parser.add_argument("--model", default="best.pt", help="Path to trained model")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time detection")
    parser.add_argument("--output", help="Output path for video")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--crop", action="store_true", help="Crop detected fish regions for classification")
    parser.add_argument("--no-show", action="store_true", help="Don't display results")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FishDetector(model_path=args.model, conf_threshold=args.conf)
    
    try:
        if args.image:
            print(f"Detecting fish in image: {args.image}")
            result = detector.detect_image(
                args.image, 
                save_result=not args.no_save,
                show_result=not args.no_show,
                crop_fish=args.crop
            )
            
            print(f"\nFound {result['total_fish']} fish in the image:")
            for det in result['detections']:
                bbox = det['bbox']
                print(f"  Fish {det['fish_id']}: {det['class_name']} (confidence: {det['confidence']:.2f})")
                print(f"    Bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            
            if args.crop and result['cropped_images']:
                print(f"\nCropped {len(result['cropped_images'])} fish regions for classification")
                
        elif args.video:
            print(f"Detecting fish in video: {args.video}")
            detections = detector.detect_video(
                args.video,
                output_path=args.output,
                show_result=not args.no_show
            )
            total_detections = sum(len(frame_dets) for frame_dets in detections)
            print(f"Total detections across all frames: {total_detections}")
            
        elif args.webcam:
            print("Starting webcam detection...")
            detector.detect_webcam()
            
        else:
            print("Please specify --image, --video, or --webcam")
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
