import os
import json
import logging
from neo4j import GraphDatabase
from RelTR.inference import load_model, predict
from sklearn.metrics import roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import psutil
import torch
import torch.serialization
import argparse
import socket
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import traceback

# Cấu hình logging để ghi lại thông tin và lỗi
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reltr_evaluation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_neo4j_connection(uri, username, password):
    """Kiểm tra kết nối đến Neo4j database"""
    try:
        driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )
        with driver.session() as session:
            result = session.run("RETURN 'Hello, Neo4j!'")
            logger.info("Successfully connected to Neo4j database")
            return True
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}")
        return False

def check_neo4j_server(host="b40b4f2a.databases.neo4j.io", port=7687):
    """Kiểm tra xem Neo4j server có đang chạy không"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            logger.info("Neo4j server is running")
            return True
        else:
            logger.error("Neo4j server is not running")
            return False
    except Exception as e:
        logger.error(f"Error checking Neo4j server: {str(e)}")
        return False

class RelTREvaluator:
    def __init__(self, 
                 neo4j_uri="neo4j+s://b40b4f2a.databases.neo4j.io",
                 neo4j_username="neo4j",
                 neo4j_password="fpKNUXKT-4z0kQMm1nuUaiXe8p70uIebc3y3a4Z8kUA",
                 model_path='./RelTR/ckpt/fine_tune1/checkpoint0049.pth',
                 image_folder='./data/vg_focused/images'):
        """
        Khởi tạo RelTREvaluator để đánh giá mô hình RelTR.
        """
        # Cấu hình kết nối Neo4j
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password)
        )
        
        # Test kết nối
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Hello, Neo4j!'")
                logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
        
        # Kiểm tra và cấu hình GPU
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Please check your GPU installation.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
            # Đặt GPU là device mặc định
            torch.cuda.set_device(0)
            # Xóa cache GPU
            torch.cuda.empty_cache()
            # Đặt số lượng thread cho CUDA
            torch.set_num_threads(1)
            # Đặt số lượng thread cho OpenMP
            os.environ['OMP_NUM_THREADS'] = '1'
            
        logger.info(f"Using device: {self.device}")
        
        # Log thông tin GPU
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            logger.info(f"Current GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"Max GPU Memory Usage: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        
        # Load model với GPU nếu có sẵn
        try:
            # Sử dụng context manager safe_globals để cho phép load argparse.Namespace
            from torch.serialization import safe_globals
            with safe_globals([argparse.Namespace]):
                # Load model với map_location để đảm bảo chuyển sang GPU
                ckpt = torch.load(model_path, map_location=self.device)
                self.model = load_model(model_path)
            
            # Chuyển model sang GPU nếu có sẵn
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                # Đảm bảo tất cả các tham số của model đều ở GPU
                for param in self.model.parameters():
                    param.data = param.data.to(self.device)
                # Đặt model ở chế độ eval
                self.model.eval()
                # Kiểm tra xem model có thực sự ở GPU không
                logger.info(f"Model is on GPU: {next(self.model.parameters()).is_cuda}")
                logger.info(f"Model device: {next(self.model.parameters()).device}")
                logger.info(f"Model state: {self.model.training}")
                # Log thông tin về model
                logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
                logger.info(f"Model parameters on GPU: {sum(p.numel() for p in self.model.parameters() if p.is_cuda)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        self.image_folder = image_folder
        
        # Tự động xác định số lượng luồng tối ưu
        self.max_workers = self._get_optimal_workers()
        logger.info(f"Using {self.max_workers} threads")
        
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
            
        logger.info("RelTR model and database loaded successfully")

    def _get_optimal_workers(self):
        """
        Xác định số lượng luồng tối ưu dựa trên cấu hình máy.
        """
        # Lấy số core CPU
        cpu_count = multiprocessing.cpu_count()
        
        # Lấy thông tin RAM
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024 * 1024 * 1024)
        
        # Tính toán số luồng tối ưu
        if torch.cuda.is_available():
            # Nếu có GPU, giảm số luồng CPU để tránh xung đột
            optimal_workers = max(1, min(cpu_count - 2, 4))  # Để lại 2 core cho hệ thống và GPU
        else:
            optimal_workers = max(1, min(cpu_count - 1, 6))  # Để lại 1 core cho hệ thống
        
        # Điều chỉnh dựa trên RAM
        if total_memory_gb >= 16:
            optimal_workers = min(optimal_workers, 8)  # Tối đa 8 luồng cho 16GB RAM
        
        logger.info(f"CPU cores: {cpu_count}")
        logger.info(f"Total RAM: {total_memory_gb:.2f} GB")
        logger.info(f"Calculated optimal workers: {optimal_workers}")
        
        return optimal_workers

    def _get_optimal_batch_size(self):
        """
        Xác định kích thước batch tối ưu dựa trên RAM và GPU.
        """
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024 * 1024 * 1024)
        
        if torch.cuda.is_available():
            # Lấy thông tin GPU
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
            # Ước tính RAM cần cho mỗi ảnh (khoảng 300MB khi dùng GPU)
            memory_per_image = 0.3  # GB
            # Để lại 2GB cho hệ thống và GPU
            available_memory = min(total_memory_gb - 2, gpu_memory - 2)
        else:
            # Ước tính RAM cần cho mỗi ảnh (khoảng 500MB khi dùng CPU)
            memory_per_image = 0.5  # GB
            # Để lại 4GB cho hệ thống
            available_memory = total_memory_gb - 4
        
        optimal_batch_size = int(available_memory / memory_per_image)
        
        # Giới hạn batch size để tránh quá tải
        return min(optimal_batch_size, 40)  # Tối đa 40 ảnh mỗi batch

    def _get_ground_truth(self, image_id):
        """
        Truy vấn mối quan hệ thực tế từ cơ sở dữ liệu Neo4j cho một ảnh.
        
        Args:
            image_id (str): ID của ảnh
        
        Returns:
            list: Các mối quan hệ thực tế trong ảnh
        """
        with self.driver.session() as session:
            query = """
            MATCH (s:Object)-[r:RELATIONSHIP]->(o:Object)
            WHERE s.image_id = $image_id
            RETURN s.category as subject, r.type as relation, o.category as object
            """
            result = session.run(query, {'image_id': image_id})
            
            ground_truth = [(record['subject'], record['relation'], record['object']) for record in result]
            return ground_truth

    def _get_predictions(self, image_id):
        """
        Predict relationships from RelTR model for an image.
        
        Args:
            image_id (str): Image ID
        
        Returns:
            list: List of tuples (subject, relation, object, confidence)
        """
        try:
            image_path = os.path.join(self.image_folder, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                logger.error(f"Image {image_id}.jpg not found in folder {self.image_folder}.")
                return None
            
            # Get predictions using the model
            pred_logits, pred_boxes, scores = self.predict(image_path)
            
            # Get max confidence scores and corresponding indices
            max_scores, pred_classes = scores.max(dim=-1)
            
            # Convert predictions to list of dictionaries with confidence
            predictions = []
            for i in range(len(max_scores)):
                if max_scores[i] > 0.3:  # Only include predictions with confidence > 0.3
                    subject_class = CLASSES[pred_classes[i][0].item()]
                    relation_class = REL_CLASSES[pred_classes[i][1].item()]
                    object_class = CLASSES[pred_classes[i][2].item()]
                    confidence = max_scores[i].item()
                    
                    predictions.append({
                        'subject': {'class': subject_class},
                        'relation': {'class': relation_class},
                        'object': {'class': object_class},
                        'confidence': confidence
                    })
            
            logger.info(f"Found {len(predictions)} predictions for image {image_id}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions for image {image_id}: {str(e)}")
            return None

    def _evaluate_relations(self, ground_truth, predictions):
        """
        Đánh giá mô hình bằng cách so sánh các mối quan hệ dự đoán và thực tế.
        
        Args:
            ground_truth (list): Các mối quan hệ thực tế
            predictions (list): Các mối quan hệ dự đoán
        
        Returns:
            dict: Các chỉ số đánh giá (TP, FP, FN, Precision, Recall, F1 Score)
        """
        # Chuyển đổi thành set để dễ dàng so sánh
        predicted_set = set(predictions)
        ground_truth_set = set(ground_truth)
        
        # Tính toán các metrics
        TP = len(predicted_set.intersection(ground_truth_set))
        FP = len(predicted_set - ground_truth_set)
        FN = len(ground_truth_set - predicted_set)
        
        # Tính precision, recall và F1 score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }

    def evaluate_image(self, image_id):
        """
        Đánh giá một ảnh cụ thể dựa trên các mối quan hệ dự đoán và thực tế.
        
        Args:
            image_id (str): ID của ảnh cần đánh giá
            
        Returns:
            dict: Kết quả đánh giá cho ảnh
        """
        try:
            ground_truth = self._get_ground_truth(image_id)
            predictions = self._get_predictions(image_id)

            evaluation_metrics = self._evaluate_relations(ground_truth, predictions)
            
            logger.info(f"Kết quả đánh giá cho ảnh {image_id}:")
            logger.info(f"True Positives (TP): {evaluation_metrics['TP']}")
            logger.info(f"False Positives (FP): {evaluation_metrics['FP']}")
            logger.info(f"False Negatives (FN): {evaluation_metrics['FN']}")
            logger.info(f"Precision: {evaluation_metrics['Precision']:.2f}")
            logger.info(f"Recall: {evaluation_metrics['Recall']:.2f}")
            logger.info(f"F1 Score: {evaluation_metrics['F1 Score']:.2f}")

            return {
                'image_id': image_id,
                'metrics': evaluation_metrics,
                'ground_truth': ground_truth,
                'predictions': predictions
            }
                
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá ảnh {image_id}: {str(e)}")
            raise

    def evaluate_multiple_images(self, image_ids):
        """
        Đánh giá nhiều ảnh và tính toán các chỉ số trung bình.
        
        Args:
            image_ids (list): Danh sách các ảnh cần đánh giá
            
        Returns:
            dict: Kết quả đánh giá tổng hợp
        """
        results = []
        total_metrics = defaultdict(float)

        for image_id in image_ids:
            try:
                result = self.evaluate_image(image_id)
                results.append(result)
                
                # Tổng hợp các số liệu
                for key, value in result['metrics'].items():
                    if isinstance(value, (int, float)):
                        total_metrics[key] += value
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá ảnh {image_id}: {str(e)}")
                continue
        
        # Tính toán số liệu trung bình
        num_images = len(results)
        if num_images > 0:
            avg_metrics = {k: v/num_images for k, v in total_metrics.items()}

            logger.info("\nKết quả đánh giá tổng hợp:")
            logger.info(f"Số lượng ảnh đã đánh giá: {num_images}")
            logger.info(f"Trung bình Precision: {avg_metrics['Precision']:.2f}")
            logger.info(f"Trung bình Recall: {avg_metrics['Recall']:.2f}")
            logger.info(f"Trung bình F1 Score: {avg_metrics['F1 Score']:.2f}")

            return {
                'individual_results': results,
                'aggregated_metrics': avg_metrics,
                'total_images_evaluated': num_images
            }
        else:
            return {
                'error': 'Không có ảnh nào được đánh giá thành công'
            }

    def query_images_by_pairs_count(self, predictions, min_pairs):
        """
        Truy vấn các ảnh có ít nhất min_pairs cặp subject-object khớp.
        """
        image_details = []
        processed_image_ids = set()
        
        with self.driver.session() as session:
            subjects = set(p['subject']['class'] for p in predictions)
            objects = set(p['object']['class'] for p in predictions)
            prediction_pairs = [(p['subject']['class'], p['object']['class']) for p in predictions]
            
            query = """
            MATCH (s:Object)
            WHERE s.category IN $subjects
            WITH DISTINCT s.image_id AS image_id
            
            MATCH (o:Object)
            WHERE o.image_id = image_id AND o.category IN $objects
            WITH DISTINCT image_id
            
            MATCH (s2:Object)-[r2:RELATIONSHIP]->(o2:Object)
            WHERE s2.image_id = image_id
            WITH image_id,
                 COLLECT(DISTINCT [s2.category, o2.category]) as image_pairs,
                 COLLECT(DISTINCT {subject: s2.category, relation: r2.type, object: o2.category}) as relationships
            
            WITH image_id, image_pairs, relationships,
                 SIZE([p IN image_pairs WHERE p IN $prediction_pairs]) as matching_pairs,
                 SIZE(image_pairs) as total_pairs
            
            WHERE matching_pairs >= $min_pairs
            RETURN DISTINCT 
                image_id, 
                matching_pairs,
                relationships,
                total_pairs,
                100.0 * matching_pairs / total_pairs as matching_percentage
            ORDER BY matching_percentage DESC, matching_pairs DESC, image_id
            """
            
            params = {
                'subjects': list(subjects),
                'objects': list(objects),
                'prediction_pairs': prediction_pairs,
                'min_pairs': min_pairs
            }
            
            result = session.run(query, params)
            
            for record in result:
                image_id = record['image_id']
                if image_id not in processed_image_ids:
                    processed_image_ids.add(image_id)
                    relationships = []
                    for rel in record['relationships']:
                        relationships.append(
                            f"{rel['subject']} -[{rel['relation']}]-> {rel['object']}"
                        )
                    
                    image_details.append({
                        "image_id": image_id,
                        "matching_pairs": record['matching_pairs'],
                        "total_pairs": record['total_pairs'],
                        "matching_percentage": round(record['matching_percentage'], 2),
                        "relationships": relationships
                    })
        
        return image_details

    def query_images_by_full_pairs_count(self, predictions, min_pairs):
        """
        Truy vấn các ảnh có ít nhất min_pairs triplets khớp.
        """
        image_details = []
        processed_image_ids = set()
        
        with self.driver.session() as session:
            subjects = set(p['subject']['class'] for p in predictions)
            relations = set(p['relation']['class'] for p in predictions)
            objects = set(p['object']['class'] for p in predictions)
            prediction_triples = [(p['subject']['class'], p['relation']['class'], p['object']['class']) for p in predictions]
            
            query = """
            MATCH (s:Object)
            WHERE s.category IN $subjects
            WITH DISTINCT s.image_id AS image_id
            
            MATCH (o:Object)
            WHERE o.image_id = image_id AND o.category IN $objects
            WITH DISTINCT image_id
            
            MATCH (s2:Object)-[r2:RELATIONSHIP]->(o2:Object)
            WHERE s2.image_id = image_id
            WITH image_id,
                 COLLECT(DISTINCT [s2.category, r2.type, o2.category]) as image_triples,
                 COLLECT(DISTINCT {subject: s2.category, relation: r2.type, object: o2.category}) as relationships
            
            WITH image_id, image_triples, relationships,
                 SIZE([t IN image_triples WHERE t IN $prediction_triples]) as matching_triples,
                 SIZE(image_triples) as total_triples
            
            WHERE matching_triples >= $min_pairs
            RETURN DISTINCT 
                image_id, 
                matching_triples,
                relationships,
                total_triples,
                100.0 * matching_triples / total_triples as matching_percentage
            ORDER BY matching_percentage DESC, matching_triples DESC, image_id
            """
            
            params = {
                'subjects': list(subjects),
                'objects': list(objects),
                'prediction_triples': prediction_triples,
                'min_pairs': min_pairs
            }
            
            result = session.run(query, params)
            
            for record in result:
                image_id = record['image_id']
                if image_id not in processed_image_ids:
                    processed_image_ids.add(image_id)
                    relationships = []
                    for rel in record['relationships']:
                        relationships.append(
                            f"{rel['subject']} -[{rel['relation']}]-> {rel['object']}"
                        )
                    
                    image_details.append({
                        "image_id": image_id,
                        "matching_triples": record['matching_triples'],
                        "total_triples": record['total_triples'],
                        "matching_percentage": round(record['matching_percentage'], 2),
                        "relationships": relationships
                    })
        
        return image_details

    def calculate_roc_metrics(self, predictions, min_pairs_range=range(1, 11)):
        """
        Tính toán ROC curve cho các ngưỡng min_pairs khác nhau.
        """
        results = {
            'pairs': [],
            'triplets': []
        }
        
        # Thêm các ngưỡng khác nhau để có nhiều điểm trên đường cong
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for min_pairs in min_pairs_range:
            # Đánh giá dựa trên cặp subject-object
            pairs_results = self.query_images_by_pairs_count(predictions, min_pairs)
            pairs_metrics = self._calculate_metrics_with_thresholds(pairs_results, thresholds)
            results['pairs'].append({
                'min_pairs': min_pairs,
                'metrics': pairs_metrics
            })
            
            # Đánh giá dựa trên triplets
            triplets_results = self.query_images_by_full_pairs_count(predictions, min_pairs)
            triplets_metrics = self._calculate_metrics_with_thresholds(triplets_results, thresholds)
            results['triplets'].append({
                'min_pairs': min_pairs,
                'metrics': triplets_metrics
            })
        
        return results

    def _calculate_metrics_with_thresholds(self, results, thresholds):
        """
        Calculate metrics for different confidence thresholds.
        
        Args:
            results (list): List of results from query_images_by_pairs_count
            thresholds (list): List of confidence thresholds to evaluate
        
        Returns:
            dict: Dictionary containing TPR and FPR for each threshold
        """
        metrics = {
            'thresholds': [],
            'tpr': [],
            'fpr': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'total_images': 0
        }
        
        for threshold in thresholds:
            # Filter predictions by confidence threshold
            filtered_results = []
            for result in results:
                filtered_preds = [
                    (p['subject']['class'], p['relation']['class'], p['object']['class'])
                    for p in result['predictions']
                    if p['confidence'] >= threshold
                ]
                if filtered_preds:  # Only include results with predictions above threshold
                    filtered_results.append({
                        'predictions': filtered_preds,
                        'ground_truth': result['ground_truth']
                    })
            
            if not filtered_results:
                continue
            
            # Calculate TP, FP, FN across all filtered results
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            for result in filtered_results:
                pred_set = set(result['predictions'])
                gt_set = set(result['ground_truth'])
                
                tp = len(pred_set.intersection(gt_set))
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            # Calculate metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # For ROC curve
            tpr = recall  # TPR is same as recall
            fpr = total_fp / (total_fp + total_fn) if (total_fp + total_fn) > 0 else 0  # FPR calculation
            
            metrics['thresholds'].append(threshold)
            metrics['tpr'].append(tpr)
            metrics['fpr'].append(fpr)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1_score'].append(f1_score)
            metrics['total_images'] = len(filtered_results)
        
        return metrics

    def plot_roc_curves(self, results, output_file='roc_curves.png'):
        """
        Vẽ ROC curves và Precision-Recall curves cho các kết quả đánh giá.
        """
        plt.figure(figsize=(20, 15))
        
        # Tạo subplot cho ROC curves
        plt.subplot(2, 2, 1)
        plt.title('ROC Curves for Subject-Object Pairs', fontsize=12)
        
        # Vẽ đường chéo 45 độ
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # Vẽ ROC curve cho cặp subject-object
        for result in results['pairs']:
            min_pairs = result['min_pairs']
            metrics = result['metrics']
            if isinstance(metrics['fpr'], (list, np.ndarray)) and isinstance(metrics['tpr'], (list, np.ndarray)):
                if len(metrics['fpr']) > 0 and len(metrics['tpr']) > 0:
                    auc = np.trapezoid(metrics['tpr'], metrics['fpr'])
                    plt.plot(metrics['fpr'], metrics['tpr'], 'o-', 
                            label=f'Min {min_pairs} pairs (AUC={auc:.3f})')
        
        plt.xlabel('False Positive Rate', fontsize=10)
        plt.ylabel('True Positive Rate', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True)
        
        # Tạo subplot cho ROC curves của triplets
        plt.subplot(2, 2, 2)
        plt.title('ROC Curves for Triplets', fontsize=12)
        
        # Vẽ đường chéo 45 độ
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # Vẽ ROC curve cho triplets
        for result in results['triplets']:
            min_pairs = result['min_pairs']
            metrics = result['metrics']
            if isinstance(metrics['fpr'], (list, np.ndarray)) and isinstance(metrics['tpr'], (list, np.ndarray)):
                if len(metrics['fpr']) > 0 and len(metrics['tpr']) > 0:
                    auc = np.trapezoid(metrics['tpr'], metrics['fpr'])
                    plt.plot(metrics['fpr'], metrics['tpr'], 'o-', 
                            label=f'Min {min_pairs} triplets (AUC={auc:.3f})')
        
        plt.xlabel('False Positive Rate', fontsize=10)
        plt.ylabel('True Positive Rate', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True)
        
        # Tạo subplot cho Precision-Recall của cặp subject-object
        plt.subplot(2, 2, 3)
        plt.title('Precision-Recall for Subject-Object Pairs', fontsize=12)
        
        for result in results['pairs']:
            min_pairs = result['min_pairs']
            metrics = result['metrics']
            if isinstance(metrics['recall'], (list, np.ndarray)) and isinstance(metrics['precision'], (list, np.ndarray)):
                if len(metrics['recall']) > 0 and len(metrics['precision']) > 0:
                    f1 = np.mean(metrics['f1_score']) if isinstance(metrics['f1_score'], (list, np.ndarray)) else metrics['f1_score']
                    plt.plot(metrics['recall'], metrics['precision'], 'o-',
                            label=f'Min {min_pairs} pairs (F1={f1:.3f})')
        
        plt.xlabel('Recall', fontsize=10)
        plt.ylabel('Precision', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True)
        
        # Tạo subplot cho Precision-Recall của triplets
        plt.subplot(2, 2, 4)
        plt.title('Precision-Recall for Triplets', fontsize=12)
        
        for result in results['triplets']:
            min_pairs = result['min_pairs']
            metrics = result['metrics']
            if isinstance(metrics['recall'], (list, np.ndarray)) and isinstance(metrics['precision'], (list, np.ndarray)):
                if len(metrics['recall']) > 0 and len(metrics['precision']) > 0:
                    f1 = np.mean(metrics['f1_score']) if isinstance(metrics['f1_score'], (list, np.ndarray)) else metrics['f1_score']
                    plt.plot(metrics['recall'], metrics['precision'], 'o-',
                            label=f'Min {min_pairs} triplets (F1={f1:.3f})')
        
        plt.xlabel('Recall', fontsize=10)
        plt.ylabel('Precision', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves and Precision-Recall curves saved to {output_file}")

    def get_all_images(self):
        """
        Lấy danh sách tất cả các ảnh trong thư mục.
        """
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith('.jpg')]
        return [os.path.splitext(f)[0] for f in image_files]

    def _process_image(self, image_id):
        """
        Xử lý một ảnh và trả về dự đoán.
        """
        try:
            image_path = os.path.join(self.image_folder, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return image_id, None
                
            # Load và xử lý ảnh
            with torch.cuda.device(self.device):
                predictions = predict(image_path, self.model)
                if predictions is None or len(predictions) == 0:
                    logger.error(f"No predictions for image: {image_id}")
                    return image_id, None
                    
                # Đảm bảo tất cả tensor đều ở đúng device
                processed_predictions = []
                for pred in predictions:
                    processed_pred = {
                        'subject': {
                            'class': pred['subject']['class'],
                            'bbox': pred['subject']['bbox'].to(self.device) if isinstance(pred['subject']['bbox'], torch.Tensor) else pred['subject']['bbox'],
                            'score': pred['subject']['score'].to(self.device) if isinstance(pred['subject']['score'], torch.Tensor) else pred['subject']['score']
                        },
                        'object': {
                            'class': pred['object']['class'],
                            'bbox': pred['object']['bbox'].to(self.device) if isinstance(pred['object']['bbox'], torch.Tensor) else pred['object']['bbox'],
                            'score': pred['object']['score'].to(self.device) if isinstance(pred['object']['score'], torch.Tensor) else pred['object']['score']
                        },
                        'relation': {
                            'class': pred['relation']['class'],
                            'score': pred['relation']['score'].to(self.device) if isinstance(pred['relation']['score'], torch.Tensor) else pred['relation']['score']
                        }
                    }
                    processed_predictions.append(processed_pred)
                
                # Log thông tin về predictions
                logger.info(f"Image {image_id}: Found {len(processed_predictions)} predictions")
                for pred in processed_predictions:
                    logger.info(f"  {pred['subject']['class']} -[{pred['relation']['class']}]-> {pred['object']['class']}")
                
                return image_id, processed_predictions
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {str(e)}")
            return image_id, None

    def _batch_query_neo4j(self, predictions_batch):
        """
        Truy vấn Neo4j cho một batch dự đoán.
        """
        if not hasattr(self, 'driver'):
            logger.error("Neo4j driver not initialized")
            return [], []
            
        with self.driver.session() as session:
            # Chuẩn bị dữ liệu cho truy vấn
            subjects = set()
            objects = set()
            prediction_pairs = []
            prediction_triples = []
            
            for pred in predictions_batch:
                if pred is None:
                    continue
                subjects.add(pred['subject']['class'])
                objects.add(pred['object']['class'])
                prediction_pairs.append((pred['subject']['class'], pred['object']['class']))
                prediction_triples.append((pred['subject']['class'], pred['relation']['class'], pred['object']['class']))
            
            if not subjects or not objects:
                logger.warning("No valid predictions in batch")
                return [], []
            
            logger.info(f"Querying Neo4j with {len(subjects)} subjects and {len(objects)} objects")
            logger.info(f"Prediction pairs: {prediction_pairs}")
            logger.info(f"Prediction triples: {prediction_triples}")
            
            # Truy vấn cho cặp subject-object
            pairs_query = """
            MATCH (s:Object)
            WHERE s.category IN $subjects
            WITH DISTINCT s.image_id AS image_id
            
            MATCH (o:Object)
            WHERE o.image_id = image_id AND o.category IN $objects
            WITH DISTINCT image_id
            
            MATCH (s2:Object)-[r2:RELATIONSHIP]->(o2:Object)
            WHERE s2.image_id = image_id
            WITH image_id,
                 COLLECT(DISTINCT [s2.category, o2.category]) as image_pairs,
                 COLLECT(DISTINCT {subject: s2.category, relation: r2.type, object: o2.category}) as relationships
            
            WITH image_id, image_pairs, relationships,
                 SIZE([p IN image_pairs WHERE p IN $prediction_pairs]) as matching_pairs,
                 SIZE(image_pairs) as total_pairs
            
            RETURN DISTINCT 
                image_id, 
                matching_pairs,
                relationships,
                total_pairs,
                100.0 * matching_pairs / total_pairs as matching_percentage
            ORDER BY matching_percentage DESC, matching_pairs DESC, image_id
            """
            
            # Truy vấn cho triplets
            triplets_query = """
            MATCH (s:Object)
            WHERE s.category IN $subjects
            WITH DISTINCT s.image_id AS image_id
            
            MATCH (o:Object)
            WHERE o.image_id = image_id AND o.category IN $objects
            WITH DISTINCT image_id
            
            MATCH (s2:Object)-[r2:RELATIONSHIP]->(o2:Object)
            WHERE s2.image_id = image_id
            WITH image_id,
                 COLLECT(DISTINCT [s2.category, r2.type, o2.category]) as image_triples,
                 COLLECT(DISTINCT {subject: s2.category, relation: r2.type, object: o2.category}) as relationships
            
            WITH image_id, image_triples, relationships,
                 SIZE([t IN image_triples WHERE t IN $prediction_triples]) as matching_triples,
                 SIZE(image_triples) as total_triples
            
            RETURN DISTINCT 
                image_id, 
                matching_triples,
                relationships,
                total_triples,
                100.0 * matching_triples / total_triples as matching_percentage
            ORDER BY matching_percentage DESC, matching_triples DESC, image_id
            """
            
            params = {
                'subjects': list(subjects),
                'objects': list(objects),
                'prediction_pairs': prediction_pairs,
                'prediction_triples': prediction_triples
            }
            
            try:
                # Thực hiện truy vấn và lưu kết quả
                pairs_result = list(session.run(pairs_query, params))
                triplets_result = list(session.run(triplets_query, params))
                
                logger.info(f"Found {len(pairs_result)} pairs and {len(triplets_result)} triplets")
                for record in pairs_result:
                    logger.info(f"Image {record['image_id']}: {record['matching_pairs']} matching pairs out of {record['total_pairs']} total pairs")
                for record in triplets_result:
                    logger.info(f"Image {record['image_id']}: {record['matching_triples']} matching triples out of {record['total_triples']} total triples")
                
                return pairs_result, triplets_result
            except Exception as e:
                logger.error(f"Error querying Neo4j: {str(e)}")
                return [], []

    def evaluate_all_images(self, batch_size=None, max_images=None, specific_images=None):
        """
        Đánh giá các ảnh trong thư mục dựa trên các mối quan hệ được dự đoán.
        Tính toán kết quả theo từng danh mục và vẽ ROC curves.
        """
        if batch_size is None:
            batch_size = self._get_optimal_batch_size()
            logger.info(f"Using optimal batch size: {batch_size}")
        
        if specific_images:
            all_images = specific_images
        else:
            all_images = self.get_all_images()
            if max_images:
                all_images = all_images[:max_images]
        
        total_images = len(all_images)
        logger.info(f"Total images to evaluate: {total_images}")
        
        # Dictionary để lưu kết quả theo danh mục
        category_results = defaultdict(lambda: {
            'total_predictions': 0,
            'correct_predictions': 0,
            'total_ground_truth': 0,
            'images': set(),
            'predictions': [],
            'ground_truth': []
        })
        
        # Dictionary để lưu kết quả cho ROC curves
        roc_results = {
            'pairs': [],
            'triplets': []
        }
        
        # Khởi tạo kết quả cho các ngưỡng
        for min_pairs in range(1, 6):
            roc_results['pairs'].append({
                'min_pairs': min_pairs,
                'metrics': {
                    'precision': [],
                    'recall': [],
                    'f1_score': [],
                    'matching_percentage': [],
                    'total_images': 0,
                    'fpr': [],
                    'tpr': []
                }
            })
            roc_results['triplets'].append({
                'min_pairs': min_pairs,
                'metrics': {
                    'precision': [],
                    'recall': [],
                    'f1_score': [],
                    'matching_percentage': [],
                    'total_images': 0,
                    'fpr': [],
                    'tpr': []
                }
            })
        
        # Xử lý ảnh theo batch với đa luồng
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(0, total_images, batch_size):
                batch_images = all_images[i:i + batch_size]
                logger.info(f"Evaluating batch {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
                
                # Xử lý ảnh song song
                future_to_image = {executor.submit(self._process_image, image_id): image_id for image_id in batch_images}
                batch_predictions = []
                
                for future in tqdm(concurrent.futures.as_completed(future_to_image), total=len(batch_images)):
                    image_id, predictions = future.result()
                    if predictions:
                        batch_predictions.extend(predictions)
                        # Lấy ground truth cho ảnh này
                        ground_truth = self._get_ground_truth(image_id)
                        
                        # Phân loại các mối quan hệ theo danh mục
                        for pred in predictions:
                            pred_category = pred['relation']['class']
                            category_results[pred_category]['total_predictions'] += 1
                            category_results[pred_category]['images'].add(image_id)
                            category_results[pred_category]['predictions'].append(pred)
                            
                            # Kiểm tra xem dự đoán có khớp với ground truth không
                            pred_tuple = (pred['subject']['class'], pred['relation']['class'], pred['object']['class'])
                            if pred_tuple in ground_truth:
                                category_results[pred_category]['correct_predictions'] += 1
                        
                        # Cập nhật tổng số ground truth cho mỗi danh mục
                        for gt in ground_truth:
                            category_results[gt[1]]['total_ground_truth'] += 1
                            category_results[gt[1]]['ground_truth'].append(gt)
                
                if batch_predictions:
                    logger.info(f"Batch {i//batch_size + 1}: Processing {len(batch_predictions)} predictions")
                    # Truy vấn Neo4j cho batch
                    pairs_result, triplets_result = self._batch_query_neo4j(batch_predictions)
                    
                    if pairs_result or triplets_result:
                        logger.info(f"Found {len(pairs_result)} pairs and {len(triplets_result)} triplets")
                        # Cập nhật kết quả ROC
                        self._update_results(roc_results, pairs_result, triplets_result)
                    else:
                        logger.warning(f"Batch {i//batch_size + 1}: No results from Neo4j query")
                else:
                    logger.warning(f"Batch {i//batch_size + 1}: No valid predictions")
                
                # Giải phóng bộ nhớ sau mỗi batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        # Tính toán và in kết quả cho từng danh mục
        logger.info("\nKết quả đánh giá theo danh mục:")
        for category, results in category_results.items():
            precision = results['correct_predictions'] / results['total_predictions'] if results['total_predictions'] > 0 else 0
            recall = results['correct_predictions'] / results['total_ground_truth'] if results['total_ground_truth'] > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            logger.info(f"\nDanh mục: {category}")
            logger.info(f"Số lượng ảnh: {len(results['images'])}")
            logger.info(f"Tổng số dự đoán: {results['total_predictions']}")
            logger.info(f"Dự đoán đúng: {results['correct_predictions']}")
            logger.info(f"Tổng số ground truth: {results['total_ground_truth']}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1_score:.4f}")
        
        # In kết quả ROC
        logger.info("\nKết quả ROC curves:")
        for metric_type in ['pairs', 'triplets']:
            for result in roc_results[metric_type]:
                metrics = result['metrics']
                logger.info(f"\nResults for {metric_type} with min_pairs={result['min_pairs']}:")
                logger.info(f"Total images: {metrics['total_images']}")
                logger.info(f"Number of data points: {len(metrics['fpr'])}")
                if len(metrics['fpr']) > 0:
                    logger.info(f"FPR range: {min(metrics['fpr']):.4f} to {max(metrics['fpr']):.4f}")
                    logger.info(f"TPR range: {min(metrics['tpr']):.4f} to {max(metrics['tpr']):.4f}")
        
        # Vẽ ROC curves
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"roc_curves_{timestamp}.png"
        self.plot_roc_curves(roc_results, output_file)
        
        return {
            'category_results': category_results,
            'roc_results': roc_results
        }

    def __del__(self):
        """
        Đóng kết nối Neo4j khi đối tượng bị hủy.
        """
        if hasattr(self, 'driver'):
            self.driver.close()
            logger.info("Neo4j connection closed")

    def _update_results(self, roc_results, pairs_result, triplets_result):
        """
        Cập nhật kết quả ROC với các cặp và triplets mới.
        """
        # Cập nhật kết quả cho cặp subject-object
        for result in roc_results['pairs']:
            min_pairs = result['min_pairs']
            metrics = result['metrics']
            
            # Lọc kết quả theo min_pairs
            filtered_pairs = [r for r in pairs_result if r['matching_pairs'] >= min_pairs]
            
            if filtered_pairs:
                # Tính toán các metrics
                total_matching = sum(r['matching_pairs'] for r in filtered_pairs)
                total_pairs = sum(r['total_pairs'] for r in filtered_pairs)
                total_images = len(filtered_pairs)
                
                precision = total_matching / total_pairs if total_pairs > 0 else 0
                recall = total_matching / (total_pairs * total_images) if total_pairs * total_images > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Cập nhật metrics
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1_score'].append(f1_score)
                metrics['total_images'] += total_images
                
                # Tính FPR và TPR
                fpr = 1 - precision
                tpr = recall
                metrics['fpr'].append(fpr)
                metrics['tpr'].append(tpr)
        
        # Cập nhật kết quả cho triplets
        for result in roc_results['triplets']:
            min_pairs = result['min_pairs']
            metrics = result['metrics']
            
            # Lọc kết quả theo min_pairs
            filtered_triplets = [r for r in triplets_result if r['matching_triples'] >= min_pairs]
            
            if filtered_triplets:
                # Tính toán các metrics
                total_matching = sum(r['matching_triples'] for r in filtered_triplets)
                total_triples = sum(r['total_triples'] for r in filtered_triplets)
                total_images = len(filtered_triplets)
                
                precision = total_matching / total_triples if total_triples > 0 else 0
                recall = total_matching / (total_triples * total_images) if total_triples * total_images > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Cập nhật metrics
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1_score'].append(f1_score)
                metrics['total_images'] += total_images
                
                # Tính FPR và TPR
                fpr = 1 - precision
                tpr = recall
                metrics['fpr'].append(fpr)
                metrics['tpr'].append(tpr)

    def predict(self, image_path):
        """
        Dự đoán các mối quan hệ trong ảnh.
        """
        try:
            # Đảm bảo model ở chế độ eval
            self.model.eval()
            
            # Load và xử lý ảnh
            image = Image.open(image_path).convert('RGB')
            transform = T.Compose([
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image)
            
            # Chuyển tensor sang device phù hợp và thêm batch dimension
            image_tensor = image_tensor.to(self.device).unsqueeze(0)
            
            # Log thông tin về tensor và device
            logger.info(f"Input tensor device: {image_tensor.device}")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            
            with torch.no_grad():
                # Thực hiện dự đoán
                outputs = self.model(image_tensor)
                
                # Xử lý kết quả
                pred_logits = outputs['pred_logits'][0]  # Remove batch dimension
                pred_boxes = outputs['pred_boxes'][0]  # Remove batch dimension
                
                # Chuyển kết quả về CPU để xử lý tiếp
                pred_logits = pred_logits.cpu()
                pred_boxes = pred_boxes.cpu()
                
                # Áp dụng softmax cho logits
                scores = F.softmax(pred_logits, dim=-1)
                
                return pred_logits, pred_boxes, scores
                
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def evaluate(self, image_ids=None):
        """
        Evaluate the model on the test set.
        
        Args:
            image_ids (list, optional): List of image IDs to evaluate. If None, evaluates on all test images.
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if image_ids is None:
            image_ids = self.test_image_ids
        
        results = []
        for image_id in tqdm(image_ids, desc="Evaluating images"):
            try:
                # Get predictions with confidence scores
                predictions = self._get_predictions(image_id)
                
                # Get ground truth
                ground_truth = self._get_ground_truth(image_id)
                
                # Store results
                results.append({
                    'predictions': predictions,
                    'ground_truth': ground_truth
                })
                
            except Exception as e:
                print(f"Error processing image {image_id}: {str(e)}")
                continue
        
        # Generate thresholds from 0 to 1
        thresholds = np.linspace(0, 1, 20)
        
        # Calculate metrics for different thresholds
        metrics = self._calculate_metrics_with_thresholds(results, thresholds)
        
        # Calculate AUC-ROC
        metrics['auc_roc'] = auc(metrics['fpr'], metrics['tpr'])
        
        # Calculate average precision
        metrics['average_precision'] = average_precision_score(
            np.array(metrics['recall']), 
            np.array(metrics['precision'])
        )
        
        return metrics

def main():
    """Hàm chính để thực hiện đánh giá"""
    try:
        # Kiểm tra Neo4j server trước
        if not check_neo4j_server():
            logger.error("Please start Neo4j server before running this script")
            return
            
        # Kiểm tra kết nối Neo4j
        if not check_neo4j_connection("neo4j+s://b40b4f2a.databases.neo4j.io", "neo4j", "fpKNUXKT-4z0kQMm1nuUaiXe8p70uIebc3y3a4Z8kUA"):
            logger.error("Cannot proceed without Neo4j connection")
            return
            
        logger.info("Initializing RelTREvaluator...")
        evaluator = RelTREvaluator()
        
        # Lấy danh sách tất cả các ảnh
        all_images = evaluator.get_all_images()
        if not all_images:
            logger.error("No images found in the image folder")
            return
            
        logger.info(f"Found {len(all_images)} images in total")
        max_images = 100  # Giới hạn số lượng ảnh để test
        test_images = all_images[:max_images]
        
        logger.info(f"Processing {len(test_images)} images")
        
        results = []
        for image_id in test_images:
            try:
                # Lấy đường dẫn ảnh
                image_path = os.path.join(evaluator.image_folder, f"{image_id}.jpg")
                if not os.path.exists(image_path):
                    logger.error(f"Image not found: {image_path}")
                    continue
                
                logger.info(f"\nProcessing image {image_id}...")
                
                # Dự đoán các mối quan hệ trong ảnh
                predictions = evaluator._get_predictions(image_id)
                if not predictions:
                    logger.error(f"No predictions for image: {image_id}")
                    continue
                
                logger.info(f"Found {len(predictions)} predictions")
                for pred in predictions:
                    logger.info(f"  {pred['subject']['class']} -[{pred['relation']['class']}]-> {pred['object']['class']} (conf: {pred['confidence']:.2f})")
                
                # Truy vấn ảnh theo cặp subject-object
                logger.info("Querying for matching pairs...")
                pairs_results = evaluator.query_images_by_pairs_count(predictions, min_pairs=1)
                logger.info(f"Found {len(pairs_results)} matching pairs")
                
                # Truy vấn ảnh theo bộ ba đầy đủ
                logger.info("Querying for matching triplets...")
                triplets_results = evaluator.query_images_by_full_pairs_count(predictions, min_pairs=1)
                logger.info(f"Found {len(triplets_results)} matching triplets")
                
                # Lưu kết quả
                result = {
                    'image_id': image_id,
                    'predictions': predictions,
                    'pairs_results': pairs_results,
                    'triplets_results': triplets_results
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing image {image_id}: {str(e)}")
                continue
        
        if not results:
            logger.error("No results were generated. Please check the logs for errors.")
            return
            
        # Lưu kết quả vào file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nEvaluation completed:")
        logger.info(f"- Total images processed: {len(results)}")
        logger.info(f"- Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    main()