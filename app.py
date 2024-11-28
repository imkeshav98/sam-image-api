import os
import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, send_from_directory
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import functools
import time
import psutil

# Setup logging
def setup_logging():
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger('sam_mask_generator')
    logger.setLevel(logging.INFO)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File Handler with log rotation
    file_handler = RotatingFileHandler(
        'logs/sam_mask_generator.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Performance and resource tracking decorator
def log_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            
            logger.info(f"Function {func.__name__} Performance:")
            logger.info(f"Execution Time: {end_time - start_time:.4f} seconds")
            logger.info(f"Memory Usage: {start_memory:.2f} MB → {end_memory:.2f} MB")
            
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

# Create necessary directories
INPUT_FOLDER = 'input_images'
OUTPUT_FOLDER = 'mask_outputs'
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Setup global logger
logger = setup_logging()

class MaskGenerator:
    def __init__(self, checkpoint_path="checkpoints/sam_vit_l_0b3195.pth"):
        """
        Initialize SAM model for mask generation
        
        :param checkpoint_path: Path to SAM model checkpoint
        """
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_sam_model()

    def _load_sam_model(self):
        """
        Load SAM model
        
        :return: Initialized SAM model
        """
        logger.info("Initializing SAM model")
        try:
            model_type = "vit_l"
            sam = sam_model_registry[model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            logger.info(f"SAM model loaded successfully on {self.device}")
            return sam
        except Exception as e:
            logger.error(f"Failed to load SAM model: {str(e)}")
            raise

    @log_performance
    def generate_masks(self, image_path, 
                       points_per_side=8, 
                       pred_iou_thresh=0.86, 
                       stability_score_thresh=0.92):
        """
        Generate masks for an input image
        
        :param image_path: Path to input image
        :param points_per_side: Number of points to sample
        :param pred_iou_thresh: Prediction IoU threshold
        :param stability_score_thresh: Stability score threshold
        :return: Tuple of original image and generated masks
        """
        logger.info(f"Generating masks for image: {image_path}")
        
        # Read image
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"Image loaded. Shape: {image.shape}")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
        
        # Create mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        # Generate masks
        try:
            masks = mask_generator.generate(image)
            logger.info(f"Generated {len(masks)} masks")
            return image, masks
        except Exception as e:
            logger.error(f"Mask generation failed: {str(e)}")
            raise

    @log_performance
    def create_mask_overlay(self, image, masks):
        """
        Create a color-coded overlay of all masks
        
        :param image: Original image
        :param masks: List of masks
        :return: Overlay image with masks
        """
        logger.info(f"Creating mask overlay for {len(masks)} masks")
        
        # Sort masks by area in descending order
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Create an overlay image with transparency
        overlay_with_masks = np.zeros((*image.shape[:2], 4), dtype=np.float32)
        
        for i, ann in enumerate(sorted_masks):
            m = ann['segmentation']
            color = np.concatenate([np.random.random(3), [0.5]])
            overlay_with_masks[m] = color
        
        logger.info("Mask overlay created successfully")
        return overlay_with_masks

    @log_performance
    def save_mask_outputs(self, filename, image, masks):
        """
        Save mask-related outputs
        
        :param filename: Original input filename
        :param image: Original image
        :param masks: Generated masks
        :return: Dictionary of output paths
        """
        base_filename = os.path.splitext(filename)[0]
        
        # Paths for outputs
        overlay_path = os.path.join(OUTPUT_FOLDER, f'{base_filename}_mask_overlay.png')
        json_path = os.path.join(OUTPUT_FOLDER, f'{base_filename}_mask_data.json')
        
        # Create and save overlay image
        overlay_with_masks = self.create_mask_overlay(image, masks)
        plt.imsave(overlay_path, overlay_with_masks)
        
        # Prepare and save mask data
        mask_data = []
        for i, mask in enumerate(masks):
            # Create individual mask image
            mask_image = np.zeros(image.shape[:2], dtype=np.uint8)
            mask_image[mask['segmentation']] = 255
            
            # Save individual mask
            mask_filename = os.path.join(OUTPUT_FOLDER, f'{base_filename}_mask_{i}.png')
            cv2.imwrite(mask_filename, mask_image)
            
            # Prepare mask entry
            mask_entry = {
                'id': i,
                'area': int(mask['area']),
                'bbox': [float(x) for x in mask['bbox']],
                'point_coords': mask['point_coords'][0],
                'crop_box': [float(x) for x in mask['crop_box']],
                'predicted_iou': float(mask['predicted_iou']),
                'stability_score': float(mask['stability_score']),
                'mask_image': f'{base_filename}_mask_{i}.png'
            }
            mask_data.append(mask_entry)
        
        # Save mask data to JSON
        with open(json_path, 'w') as f:
            json.dump(mask_data, f, indent=4)
        
        return {
            "overlay_path": overlay_path,
            "json_path": json_path,
            "mask_images": [f'{base_filename}_mask_{i}.png' for i in range(len(masks))],
        }

# Flask Application
app = Flask(__name__)

# Initialize Mask Generator
mask_generator = MaskGenerator()

@app.route('/process_image/<filename>')
def process_image(filename):
    """
    Process an image and generate masks
    
    :param filename: Name of the image file to process
    :return: JSON response with processing results
    """
    logger.info(f"Processing image: {filename}")
    
    # Full path to input image
    input_path = os.path.join(INPUT_FOLDER, filename)
    
    # Verify image exists
    if not os.path.exists(input_path):
        logger.warning(f"Image not found: {input_path}")
        return jsonify({"error": "Image not found"}), 404
    
    try:
        # Generate masks
        image, masks = mask_generator.generate_masks(input_path)
        
        # Save outputs
        outputs = mask_generator.save_mask_outputs(filename, image, masks)
        
        logger.info("Image processing completed successfully")
        return jsonify({
            "message": "Image processed successfully",
            **outputs,
            "num_masks": len(masks)
        })
    
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/list_images')
def list_images():
    """
    List available images in the input folder
    
    :return: JSON response with list of image filenames
    """
    logger.info("Listing available images")
    images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    logger.info(f"Found {len(images)} images")
    return jsonify({"images": images})

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """
    Serve output files
    
    :param filename: Name of the output file
    :return: File download
    """
    return send_from_directory(OUTPUT_FOLDER, filename)

# Error handler
@app.errorhandler(500)
def handle_500(error):
    logger.error(f"Internal Server Error: {str(error)}", exc_info=True)
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    logger.info("Starting SAM Mask Generator Flask API")
    app.run(debug=True, host='0.0.0.0', port=5000)