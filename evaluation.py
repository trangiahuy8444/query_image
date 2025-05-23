import os
import json
import numpy as np
from test import load_model_and_predict, get_predictions_from_model, query_images_by_pairs_parallel, query_images_triplets_parallel, calculate_roc_pr_metrics
from sklearn.metrics import precision_recall_fscore_support

def evaluate_model_pipeline(input_path, model_path, save_dir="./evaluation_results", max_images=None):
    """
    Evaluate model predictions and generate metrics for a single image or folder of images
    
    Args:
        input_path: Path to single image or folder containing images
        model_path: Path to model checkpoint
        save_dir: Directory to save evaluation results
        max_images: Maximum number of images to evaluate (None for all images)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle single image or folder of images
    if os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Limit the number of images if max_images is specified
        if max_images is not None and max_images < len(image_paths):
            print(f"Limiting evaluation to {max_images} images out of {len(image_paths)} found")
            image_paths = image_paths[:max_images]
    
    print(f"Evaluating {len(image_paths)} images")
    
    all_results = {}
    
    # Dictionaries to store aggregated metrics for precision, recall, and F1
    aggregated_metrics = {
        'pairs': {f'min_{i}': {'y_true': [], 'y_score': []} for i in range(1, 5)},
        'triplets': {f'min_{i}': {'y_true': [], 'y_score': []} for i in range(1, 5)}
    }
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
        
        # Step 1: Get model predictions
        predictions = load_model_and_predict(image_path, model_path)
        model_predictions = get_predictions_from_model(predictions)
        
        if not model_predictions:
            print(f"No valid predictions for {image_path}")
            continue
            
        # Step 2: Get ground truth data
        ground_truth_pairs = []
        ground_truth_triplets = []
        
        # Query with different min_pairs values (1-4)
        for min_pairs in range(1, 5):
            pairs = query_images_by_pairs_parallel([predictions], min_pairs)
            triplets = query_images_triplets_parallel([predictions], min_pairs)
            
            if pairs:
                ground_truth_pairs.extend(pairs)
                print(f"Found {len(pairs)} pairs for min_pairs={min_pairs}")
            if triplets:
                ground_truth_triplets.extend(triplets)
                print(f"Found {len(triplets)} triplets for min_pairs={min_pairs}")
        
        print(f"Total ground truth pairs: {len(ground_truth_pairs)}")
        print(f"Total ground truth triplets: {len(ground_truth_triplets)}")
        
        # Step 3: Calculate metrics
        pairs_metrics = {}
        triplets_metrics = {}
        
        # Calculate metrics for pairs (1-4)
        for min_pairs in range(1, 5):
            pairs_data = [p for p in ground_truth_pairs if p.get('matching_pairs', 0) >= min_pairs]
            print(f"Pairs data for min_pairs={min_pairs}: {len(pairs_data)} items")
            
            if pairs_data:
                y_true, y_score = calculate_roc_pr_metrics(model_predictions, pairs_data)
                pairs_metrics[f'min_{min_pairs}'] = {
                    'y_true': y_true.tolist(),
                    'y_score': y_score.tolist()
                }
                print(f"Calculated metrics for pairs min_pairs={min_pairs}: y_true={len(y_true)}, y_score={len(y_score)}")
                
                # Add to aggregated metrics
                aggregated_metrics['pairs'][f'min_{min_pairs}']['y_true'].extend(y_true.tolist())
                aggregated_metrics['pairs'][f'min_{min_pairs}']['y_score'].extend(y_score.tolist())
            else:
                # Create empty metrics to ensure all curves are plotted
                pairs_metrics[f'min_{min_pairs}'] = {
                    'y_true': [0, 1],  # At least one positive and one negative
                    'y_score': [0.1, 0.9]  # Different scores to create a curve
                }
                print(f"No data for pairs min_pairs={min_pairs}, using placeholder")
        
        # Calculate metrics for triplets (1-4)
        for min_triplets in range(1, 5):
            triplets_data = [t for t in ground_truth_triplets if t.get('matching_triples', 0) >= min_triplets]
            print(f"Triplets data for min_triplets={min_triplets}: {len(triplets_data)} items")
            
            if triplets_data:
                y_true, y_score = calculate_roc_pr_metrics(model_predictions, triplets_data)
                triplets_metrics[f'min_{min_triplets}'] = {
                    'y_true': y_true.tolist(),
                    'y_score': y_score.tolist()
                }
                print(f"Calculated metrics for triplets min_triplets={min_triplets}: y_true={len(y_true)}, y_score={len(y_score)}")
                
                # Add to aggregated metrics
                aggregated_metrics['triplets'][f'min_{min_triplets}']['y_true'].extend(y_true.tolist())
                aggregated_metrics['triplets'][f'min_{min_triplets}']['y_score'].extend(y_score.tolist())
            else:
                # Create empty metrics to ensure all curves are plotted
                triplets_metrics[f'min_{min_triplets}'] = {
                    'y_true': [0, 1],  # At least one positive and one negative
                    'y_score': [0.1, 0.9]  # Different scores to create a curve
                }
                print(f"No data for triplets min_triplets={min_triplets}, using placeholder")
        
        # Store results
        image_name = os.path.basename(image_path)
        all_results[image_name] = {
            'pairs_metrics': pairs_metrics,
            'triplets_metrics': triplets_metrics
        }
    
    # Save all metrics to JSON
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nEvaluation metrics saved to {os.path.join(save_dir, 'evaluation_metrics.json')}")
    
    # Calculate and save average precision, recall, and F1 scores
    average_metrics = calculate_average_metrics(aggregated_metrics)
    
    with open(os.path.join(save_dir, 'average_metrics.json'), 'w') as f:
        json.dump(average_metrics, f, indent=4)
    
    print(f"Average metrics saved to {os.path.join(save_dir, 'average_metrics.json')}")
    
    return all_results

def calculate_average_metrics(aggregated_metrics):
    """
    Calculate average precision, recall, and F1 scores from aggregated metrics
    
    Args:
        aggregated_metrics: Dictionary containing aggregated y_true and y_score for each category
        
    Returns:
        Dictionary containing average metrics for each category
    """
    average_metrics = {
        'pairs': {},
        'triplets': {}
    }
    
    # Calculate metrics for pairs
    for min_key in aggregated_metrics['pairs']:
        y_true = np.array(aggregated_metrics['pairs'][min_key]['y_true'])
        y_score = np.array(aggregated_metrics['pairs'][min_key]['y_score'])
        
        if len(y_true) > 0 and len(y_score) > 0:
            # Convert scores to binary predictions using 0.5 as threshold
            y_pred = (y_score >= 0.5).astype(int)
            
            # Calculate precision, recall, and F1
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            
            average_metrics['pairs'][min_key] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'sample_count': len(y_true)
            }
        else:
            average_metrics['pairs'][min_key] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'sample_count': 0
            }
    
    # Calculate metrics for triplets
    for min_key in aggregated_metrics['triplets']:
        y_true = np.array(aggregated_metrics['triplets'][min_key]['y_true'])
        y_score = np.array(aggregated_metrics['triplets'][min_key]['y_score'])
        
        if len(y_true) > 0 and len(y_score) > 0:
            # Convert scores to binary predictions using 0.5 as threshold
            y_pred = (y_score >= 0.5).astype(int)
            
            # Calculate precision, recall, and F1
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            
            average_metrics['triplets'][min_key] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'sample_count': len(y_true)
            }
        else:
            average_metrics['triplets'][min_key] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'sample_count': 0
            }
    
    return average_metrics

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate model predictions on images')
    parser.add_argument('--input_path', type=str, default="data/vg_focused/images", 
                        help='Path to single image or folder containing images')
    parser.add_argument('--model_path', type=str, default="RelTR/ckpt/fine_tune1/checkpoint0049.pth", 
                        help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, default="evaluation_results_full_ft", 
                        help='Directory to save evaluation results')
    parser.add_argument('--max_images', type=int, default=None, 
                        help='Maximum number of images to evaluate (default: all images)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model_pipeline(
        input_path=args.input_path,
        model_path=args.model_path,
        save_dir=args.save_dir,
        max_images=args.max_images
    ) 