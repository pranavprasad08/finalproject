import json
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from skimage import measure
from pycocotools import mask as maskUtils
import cv2
import matplotlib.pyplot as plt

class CustomCOCOEvaluator:
    def __init__(self, gt_annotations_path, pred_json_path, iou_threshold=0.5):
        """
        Initialize the evaluator with paths to ground truth and prediction annotations.
        Args:
            gt_annotations_path (str): Path to the ground truth annotations (COCO format).
            pred_json_path (str): Path to the predictions in COCO format.
            iou_threshold (float): IoU threshold to consider a detection as a true positive.
        """
        self.gt_annotations_path = gt_annotations_path
        self.pred_json_path = pred_json_path
        self.iou_threshold = iou_threshold
        
        # Load the annotation and prediction data from JSON files
        self.gt_data = self.load_json(gt_annotations_path)
        self.pred_data = self.load_json(pred_json_path)
        
        # Organize ground truth and predictions by image ID
        self.gt_dict = self.organize_by_image_id(self.gt_data['annotations'])
        self.pred_dict = self.organize_by_image_id(self.pred_data)
        
        # Extract image sizes (height, width) for each image ID
        self.image_sizes = self.get_image_sizes(self.gt_data['images'])
        
        # Create a mapping from category IDs to category names
        self.category_map = self.create_category_map(self.gt_data['categories'])

    def get_image_sizes(self, images):
        """
        Extract the height and width of each image from the ground truth data.
        Args:
            images (list): List of image metadata dictionaries from the ground truth annotations.
        Returns:
            dict: Dictionary mapping image IDs to (height, width) tuples.
        """
        image_sizes = {}
        for img in images:
            image_sizes[img['id']] = (img['height'], img['width'])
        return image_sizes

    def load_json(self, path):
        with open(path) as f:
            return json.load(f)

    def organize_by_image_id(self, annotations):
        organized = {}
        for ann in annotations:
            image_id = ann['image_id']
            if image_id not in organized:
                organized[image_id] = []
            organized[image_id].append(ann)
        return organized

    def create_category_map(self, categories):
        category_map = {}
        for cat in categories:
            category_map[cat['id']] = cat['name']
        return category_map

    def compute_iou_bbox(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h2, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

    def rle_to_mask(self, rle, height, width):
        """Convert RLE to a binary mask."""
        return maskUtils.decode(rle).astype(np.uint8)

    def polygon_to_mask(self, polygons, height, width):
        """
        Convert polygon coordinates to a binary mask.
        Args:
            polygons (list): List of polygons (each polygon is a list of coordinates).
            height (int): Height of the image.
            width (int): Width of the image.
        Returns:
            np.ndarray: Binary mask.
        """
        # Ensure height and width are not None
        if height is None or width is None:
            raise ValueError(f"Invalid image dimensions: height={height}, width={width}")
    
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill the mask with each polygon
        for polygon in polygons:
            contours = np.array(polygon).reshape(-1, 2)
            mask = cv2.fillPoly(mask, [contours.astype(int)], 1)
        
        return mask

    def compute_iou_mask(self, gt, pred, height, width):
        """
        Compute the IoU between two segmentation masks.
        Args:
            gt (dict): Ground truth annotation with 'segmentation'.
            pred (dict): Prediction annotation with 'segmentation'.
            height (int): Height of the image.
            width (int): Width of the image.
        Returns:
            float: IoU value.
        """
        # Convert the ground truth and predicted segmentations to binary masks
        if isinstance(gt['segmentation'], list):  # Polygon format
            gt_mask = self.polygon_to_mask(gt['segmentation'], height, width)
        elif isinstance(gt['segmentation'], dict):  # RLE format
            gt_mask = self.rle_to_mask(gt['segmentation'], height, width)
        else:
            raise ValueError("Unsupported segmentation format for ground truth")
    
        if isinstance(pred['segmentation'], list):  # Polygon format
            pred_mask = self.polygon_to_mask(pred['segmentation'], height, width)
        elif isinstance(pred['segmentation'], dict):  # RLE format
            pred_mask = self.rle_to_mask(pred['segmentation'], height, width)
        else:
            raise ValueError("Unsupported segmentation format for prediction")
    
        gt_poly = self.mask_to_polygon(gt_mask)
        pred_poly = self.mask_to_polygon(pred_mask)
    
        if gt_poly.is_empty or pred_poly.is_empty:
            return 0.0
    
        try:
            # Compute intersection and union areas using unary_union
            intersection = gt_poly.intersection(pred_poly).area
            union = unary_union([gt_poly, pred_poly]).area
        except Exception as e:  # Catching general exceptions
            print(f"Skipping IoU computation due to exception: {e}")
            return 0.0
    
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        return iou


    def mask_to_polygon(self, mask):
        """
        Convert a binary mask to a polygon using contour detection.
        Args:
            mask (np.ndarray): Binary mask.
        Returns:
            Polygon or MultiPolygon: Polygon representing the mask.
        """
        contours = measure.find_contours(mask, 0.5)
        polygons = [Polygon(contour) for contour in contours if len(contour) >= 3]
        
        if polygons:
            try:
                # Combine all polygons into a MultiPolygon using unary_union
                polygon = unary_union(polygons)
                
                # Attempt to fix the geometry
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)  # This often fixes invalid geometries
    
                # If the geometry is still invalid, return an empty MultiPolygon
                if not polygon.is_valid or polygon.is_empty:
                    print("Invalid polygon detected; skipping this geometry.")
                    return MultiPolygon()
    
                return polygon
            except Exception as e:  # Catching general exceptions
                print(f"Skipping invalid polygon due to exception: {e}")
                return MultiPolygon()
        else:
            return Polygon()
     

    def match_predictions(self, gt_items, pred_items, iou_threshold, iou_type='bbox', height=None, width=None):
        matches = []
        for pred in pred_items:
            for gt in gt_items:
                if iou_type == 'bbox':
                    iou = self.compute_iou_bbox(gt['bbox'], pred['bbox'])
                elif iou_type == 'segm':
                    iou = self.compute_iou_mask(gt, pred, height, width)
                if iou >= iou_threshold:
                    matches.append((gt, pred, iou))
                    break
        return matches

    def evaluate_image(self, image_id, iou_type='bbox'):
        """
        Evaluate the predictions for a single image.
        Args:
            image_id (int): ID of the image to evaluate.
            iou_type (str): Type of IoU to compute ('bbox' or 'segm').
        Returns:
            tuple: (true positives, false positives, false negatives, matches)
        """
        gt_items = self.gt_dict.get(image_id, [])
        pred_items = self.pred_dict.get(image_id, [])
        
        # Retrieve image dimensions
        dimensions = self.image_sizes.get(image_id)
        
        if dimensions is None:
            print(f"Skipping image ID {image_id}: dimensions not found")
            return 0, 0, 0, []  # Skip this image by returning zeros
        
        height, width = dimensions
        
        matches = self.match_predictions(gt_items, pred_items, self.iou_threshold, iou_type, height, width)
    
        tp = len(matches)
        fp = len(pred_items) - tp
        fn = len(gt_items) - tp
        return tp, fp, fn, matches

    def evaluate(self, iou_type='bbox'):
        """
        Evaluate the predictions for all images and compute per-class metrics.
        Args:
            iou_type (str): Type of IoU to compute ('bbox' or 'segm').
        Returns:
            tuple: (overall precision, overall recall, class metrics)
        """
        all_tp = 0
        all_fp = 0
        all_fn = 0
        all_tn = 0  # We'll assume TN is the total number of true negatives
        category_metrics = {cat_id: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'matches': []} for cat_id in self.category_map}
    
        for image_id in self.gt_dict:
            # Only pass image_id and iou_type; height and width are handled inside evaluate_image
            tp, fp, fn, matches = self.evaluate_image(image_id, iou_type)
            
            # Skip images with missing dimensions or annotations
            if tp == 0 and fp == 0 and fn == 0 and not matches:
                continue
            
            all_tp += tp
            all_fp += fp
            all_fn += fn
    
            # Aggregate metrics per category
            for gt, pred, iou in matches:
                cat_id = gt['category_id']
                category_metrics[cat_id]['tp'] += 1
                category_metrics[cat_id]['matches'].append((gt, pred, iou))
            for pred in self.pred_dict.get(image_id, []):
                if not any(m[1] == pred for m in matches):
                    category_metrics[pred['category_id']]['fp'] += 1
            for gt in self.gt_dict.get(image_id, []):
                if not any(m[0] == gt for m in matches):
                    category_metrics[gt['category_id']]['fn'] += 1
            
            # Calculate TN per image (assuming total possible negatives is based on some fixed value)
            tn = len(self.gt_dict) - (tp + fp + fn)  # Simplified; usually, tn would be based on total negatives
            all_tn += tn
    
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        accuracy = (all_tp + all_tn) / (all_tp + all_fp + all_fn + all_tn) if (all_tp + all_fp + all_fn + all_tn) > 0 else 0
        specificity = all_tn / (all_tn + all_fp) if (all_tn + all_fp) > 0 else 0
        fpr = all_fp / (all_fp + all_tn) if (all_fp + all_tn) > 0 else 0
        fnr = all_fn / (all_fn + all_tp) if (all_fn + all_tp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Overall Specificity: {specificity:.4f}")
        print(f"Overall FPR: {fpr:.4f}")
        print(f"Overall FNR: {fnr:.4f}")
        print(f"Overall F1 Score: {f1:.4f}")
    
        # Calculate per-class metrics
        class_metrics = {}
        for cat_id, metrics in category_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            tn = metrics['tn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            ap = precision * recall  # Simplified for demonstration; normally you'd integrate over recall levels
    
            class_metrics[self.category_map[cat_id]] = {
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'specificity': specificity,
                'fpr': fpr,
                'fnr': fnr,
                'f1': f1,
                'ap': ap
            }
            print(f"Class '{self.category_map[cat_id]}': Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Specificity: {specificity:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}, F1 Score: {f1:.4f}, AP: {ap:.4f}")
    
        return precision, recall, class_metrics


    def visualize_class_metrics(self, class_metrics):
        categories = list(class_metrics.keys())
        precisions = [class_metrics[cat]['precision'] for cat in categories]
        recalls = [class_metrics[cat]['recall'] for cat in categories]
        aps = [class_metrics[cat]['ap'] for cat in categories]

        x = np.arange(len(categories))

        plt.figure(figsize=(14, 8))

        plt.bar(x - 0.2, precisions, 0.2, label='Precision', color='blue')
        plt.bar(x, recalls, 0.2, label='Recall', color='green')
        plt.bar(x + 0.2, aps, 0.2, label='AP', color='red')

        plt.xlabel('Class')
        plt.ylabel('Metric Value')
        plt.title('Precision, Recall, and AP per Class')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()

        plt.tight_layout()
        plt.show()
