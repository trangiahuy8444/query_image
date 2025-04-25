import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_metrics_from_json(json_file_path, save_path=None):
    """
    Plot ROC and Precision-Recall curves from evaluation_metrics.json
    
    Args:
        json_file_path: Path to the evaluation_metrics.json file
        save_path: Path to save the plot (if None, display the plot)
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Plot ROC curves
    plt.subplot(1, 2, 1)
    
    # Plot pairs ROC curves (excluding min_5)
    for i, min_key in enumerate(['min_1', 'min_2', 'min_3', 'min_4']):
        # Collect all y_true and y_score for this min_key across all images
        all_y_true = []
        all_y_score = []
        
        for image_name, image_data in data.items():
            if min_key in image_data['pairs_metrics']:
                metrics = image_data['pairs_metrics'][min_key]
                all_y_true.extend(metrics['y_true'])
                all_y_score.extend(metrics['y_score'])
        
        # Convert to numpy arrays
        y_true = np.array(all_y_true)
        y_score = np.array(all_y_score)
        
        if len(y_true) > 1 and len(y_score) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Use different colors for different min values
            if min_key == 'min_1':
                color = 'red'
            elif min_key == 'min_2':
                color = 'red'
            elif min_key == 'min_3':
                color = 'red'
            elif min_key == 'min_4':
                color = 'cyan'
            
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'Pairs (≥{i+1}) (AUC = {roc_auc:.2f})')
            print(f"Plotted ROC curve for pairs {min_key} with {len(y_true)} data points")
    
    # Plot triplets ROC curves (excluding min_5)
    for i, min_key in enumerate(['min_1', 'min_2', 'min_3', 'min_4']):
        # Collect all y_true and y_score for this min_key across all images
        all_y_true = []
        all_y_score = []
        
        for image_name, image_data in data.items():
            if min_key in image_data['triplets_metrics']:
                metrics = image_data['triplets_metrics'][min_key]
                all_y_true.extend(metrics['y_true'])
                all_y_score.extend(metrics['y_score'])
        
        # Convert to numpy arrays
        y_true = np.array(all_y_true)
        y_score = np.array(all_y_score)
        
        if len(y_true) > 1 and len(y_score) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Use different colors for different min values
            if min_key == 'min_1':
                color = 'darkorange'
            elif min_key == 'min_2':
                color = 'darkorange'
            elif min_key == 'min_3':
                color = 'darkorange'
            elif min_key == 'min_4':
                color = 'green'
            
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'Triplets (≥{i+1}) (AUC = {roc_auc:.2f})')
            print(f"Plotted ROC curve for triplets {min_key} with {len(y_true)} data points")
    
    # Đường chuẩn
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Pairs and Triplets')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid()
    
    # Plot Precision-Recall curves
    plt.subplot(1, 2, 2)
    
    # Plot pairs PR curves (excluding min_5)
    for i, min_key in enumerate(['min_1', 'min_2', 'min_3', 'min_4']):
        # Collect all y_true and y_score for this min_key across all images
        all_y_true = []
        all_y_score = []
        
        for image_name, image_data in data.items():
            if min_key in image_data['pairs_metrics']:
                metrics = image_data['pairs_metrics'][min_key]
                all_y_true.extend(metrics['y_true'])
                all_y_score.extend(metrics['y_score'])
        
        # Convert to numpy arrays
        y_true = np.array(all_y_true)
        y_score = np.array(all_y_score)
        
        if len(y_true) > 1 and len(y_score) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            
            # Use different colors for different min values
            if min_key == 'min_1':
                color = 'red'
            elif min_key == 'min_2':
                color = 'red'
            elif min_key == 'min_3':
                color = 'red'
            elif min_key == 'min_4':
                color = 'cyan'
            
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'Pairs (≥{i+1}) (AP = {pr_auc:.2f})')
            print(f"Plotted PR curve for pairs {min_key} with {len(y_true)} data points")
    
    # Plot triplets PR curves (excluding min_5)
    for i, min_key in enumerate(['min_1', 'min_2', 'min_3', 'min_4']):
        # Collect all y_true and y_score for this min_key across all images
        all_y_true = []
        all_y_score = []
        
        for image_name, image_data in data.items():
            if min_key in image_data['triplets_metrics']:
                metrics = image_data['triplets_metrics'][min_key]
                all_y_true.extend(metrics['y_true'])
                all_y_score.extend(metrics['y_score'])
        
        # Convert to numpy arrays
        y_true = np.array(all_y_true)
        y_score = np.array(all_y_score)
        
        if len(y_true) > 1 and len(y_score) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            
            # Use different colors for different min values
            if min_key == 'min_1':
                color = 'darkorange'
            elif min_key == 'min_2':
                color = 'darkorange'
            elif min_key == 'min_3':
                color = 'darkorange'
            elif min_key == 'min_4':
                color = 'green'
            
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'Triplets (≥{i+1}) (AP = {pr_auc:.2f})')
            print(f"Plotted PR curve for triplets {min_key} with {len(y_true)} data points")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Pairs and Triplets')
    plt.legend(loc="lower left", fontsize=8)
    plt.grid()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Path to the evaluation_metrics.json file
    json_file_path = "./evaluation_results/evaluation_metrics.json"
    
    # Path to save the plot
    save_path = "./evaluation_results/combined_curves_all_images.png"
    
    # Plot the metrics
    plot_metrics_from_json(json_file_path, save_path) 