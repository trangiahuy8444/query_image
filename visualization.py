import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import json
import os
from datetime import datetime
from PIL import Image
import requests
import io
from neo4j import GraphDatabase

class QueryMetricsVisualizer:
    def __init__(self, metrics_data=None, output_dir='./visualization_output'):
        """
        Initialize the visualizer with metrics data
        
        Args:
            metrics_data: Dictionary containing metrics data from the API response
            output_dir: Directory to save visualization outputs
        """
        self.metrics = metrics_data or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Neo4j connection
        self.neo4j_uri = "neo4j+s://b40b4f2a.databases.neo4j.io"
        self.neo4j_username = "neo4j"
        self.neo4j_password = "fpKNUXKT-4z0kQMm1nuUaiXe8p70uIebc3y3a4Z8kUA"
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
    def get_total_images_count(self):
        """Get total number of images in the database"""
        with self.driver.session() as session:
            result = session.run("MATCH (n:Object) RETURN COUNT(DISTINCT n.image_id) as count")
            return result.single()["count"]
    
    def _prepare_roc_data(self):
        """Prepare data for ROC curve"""
        fpr = []  # False Positive Rate
        tpr = []  # True Positive Rate
        thresholds = []
        
        # Process pairs data
        for threshold, data in self.metrics.items():
            if 'pairs_' in threshold:
                tp = data['true_positives']
                fp = data['false_positives']
                fn = data['false_negatives']
                tn = data['true_negatives']
                
                fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
                tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                thresholds.append(threshold)
        
        return np.array(fpr), np.array(tpr), thresholds
    
    def _prepare_pr_data(self):
        """Prepare data for Precision-Recall curve"""
        precision = []
        recall = []
        thresholds = []
        
        # Process pairs data
        for threshold, data in self.metrics.items():
            if 'pairs_' in threshold:
                precision.append(data['precision'])
                recall.append(data['recall'])
                thresholds.append(threshold)
        
        return np.array(precision), np.array(recall), thresholds
    
    def plot_roc_curve(self, save=True):
        """Plot ROC curve"""
        fpr, tpr, thresholds = self._prepare_roc_data()
        if len(fpr) == 0 or len(tpr) == 0:
            print("No data available for ROC curve")
            return
            
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Add threshold points
        for i, threshold in enumerate(thresholds):
            plt.scatter(fpr[i], tpr[i], color='red')
            plt.annotate(threshold, (fpr[i], tpr[i]))
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(self.output_dir, f'roc_curve_{timestamp}.png'))
        
        plt.show()
    
    def plot_precision_recall_curve(self, save=True):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = self._prepare_pr_data()
        if len(precision) == 0 or len(recall) == 0:
            print("No data available for Precision-Recall curve")
            return
            
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                label='Precision-Recall curve')
        
        # Add threshold points
        for i, threshold in enumerate(thresholds):
            plt.scatter(recall[i], precision[i], color='red')
            plt.annotate(threshold, (recall[i], precision[i]))
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(self.output_dir, f'precision_recall_curve_{timestamp}.png'))
        
        plt.show()
    
    def plot_metrics_comparison(self, save=True):
        """Plot comparison of different query methods"""
        methods = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold, data in self.metrics.items():
            if 'pairs_' in threshold or 'full_triples_' in threshold:
                methods.append(threshold)
                precisions.append(data['precision'])
                recalls.append(data['recall'])
                f1_scores.append(data['f1_score'])
        
        if not methods:
            print("No data available for metrics comparison")
            return
            
        x = np.arange(len(methods))
        width = 0.25
        
        plt.figure(figsize=(12, 8))
        plt.bar(x - width, precisions, width, label='Precision')
        plt.bar(x, recalls, width, label='Recall')
        plt.bar(x + width, f1_scores, width, label='F1 Score')
        
        plt.xlabel('Query Methods')
        plt.ylabel('Scores')
        plt.title('Comparison of Query Methods')
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(self.output_dir, f'metrics_comparison_{timestamp}.png'))
        
        plt.show()

    def display_query_results(self, results):
        """Display the query results including images and relationships"""
        predictions = results.get('predictions', [])
        metrics = results.get('metrics', {})
        
        # Display predictions
        print("\nPredicted Relationships:")
        print("-" * 50)
        for i, pred in enumerate(predictions, 1):
            subject = pred['subject']['class']
            relation = pred['relation']['class']
            object = pred['object']['class']
            print(f"{i}. {subject} -[{relation}]-> {object}")
        
        # Display metrics
        print("\nQuery Metrics:")
        print("-" * 50)
        print(f"Prediction Time: {metrics.get('prediction_time', 0):.2f} seconds")
        print(f"Number of Predictions: {metrics.get('num_predictions', 0)}")
        print(f"Total Unique Images Found: {metrics.get('total_unique_images', 0)}")
        
        # Display ROC and PR metrics
        roc_pr_metrics = metrics.get('roc_pr_metrics', {})
        if roc_pr_metrics:
            print("\nROC and Precision-Recall Metrics:")
            for method, data in roc_pr_metrics.items():
                if 'pairs_' in method or 'full_triples_' in method:
                    print(f"\n{method}:")
                    print(f"  Precision: {data.get('precision', 0):.2f}")
                    print(f"  Recall: {data.get('recall', 0):.2f}")
                    print(f"  F1 Score: {data.get('f1_score', 0):.2f}")
                    print(f"  Total Retrieved: {data.get('total_retrieved', 0)}")

    def query_and_visualize(self, image_path, server_url='neo4j+s://b40b4f2a.databases.neo4j.io'):
        """Query an image and visualize the results"""
        try:
            # Query the image
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f'{server_url}/upload', files=files)
                
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                return
            
            results = response.json()
            
            # Get total images count from Neo4j
            total_images = self.get_total_images_count()
            print(f"\nTotal images in database: {total_images}")
            
            # Display results
            self.display_query_results(results)
            
            # Generate visualizations
            print("\nGenerating visualizations...")
            self.metrics = results.get('metrics', {}).get('roc_pr_metrics', {})
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
            self.plot_metrics_comparison()
            
            print("\nVisualizations saved to ./visualization_output directory")
            
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    # Get image path from user
    image_path = "image_test/150542.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Initialize visualizer and process the image
    visualizer = QueryMetricsVisualizer()
    visualizer.query_and_visualize(image_path)

if __name__ == '__main__':
    main() 