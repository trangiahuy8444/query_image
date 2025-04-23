import os
import json
import logging
from neo4j import GraphDatabase
from RelTR.inference import load_model, predict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import psutil
import torch
import torch.cuda
import torch.serialization
import argparse

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

class RelTREvaluator:
    def __init__(self, 
                 neo4j_uri="bolt://localhost:7687",
                 neo4j_username="neo4j",
                 neo4j_password="12345678",
                 model_path='./RelTR/ckpt/fine_tune1/checkpoint0049.pth',
                 image_folder='./data/vg_focused/images'):
        """
        Khởi tạo RelTREvaluator để đánh giá mô hình RelTR.
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        
        # Kiểm tra và sử dụng GPU nếu có
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model với GPU nếu có
        try:
            # Thêm argparse.Namespace vào danh sách safe globals
            torch.serialization.add_safe_globals([argparse.Namespace])
            # Load model với weights_only=False
            self.model = load_model(model_path)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                # Tối ưu hóa bộ nhớ GPU
                torch.cuda.empty_cache()
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
            optimal_workers = min(cpu_count - 2, 4)  # Để lại 2 core cho hệ thống và GPU
        else:
            optimal_workers = min(cpu_count - 1, 6)  # Để lại 1 core cho hệ thống
        
        # Điều chỉnh dựa trên RAM
        if total_memory_gb >= 16:
            optimal_workers = min(optimal_workers, 8)  # Tối đa 8 luồng cho 16GB RAM
        
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
        Dự đoán các mối quan hệ từ mô hình RelTR cho một ảnh.
        
        Args:
            image_id (str): ID của ảnh
        
        Returns:
            list: Các mối quan hệ dự đoán từ mô hình
        """
        image_path = os.path.join(self.image_folder, f"{image_id}.jpg")
        if not os.path.exists(image_path):  # Kiểm tra xem ảnh có tồn tại không
            raise FileNotFoundError(f"Ảnh {image_id}.jpg không tìm thấy trong thư mục {self.image_folder}.")
        predictions = predict(image_path, self.model)
        return [(p['subject']['class'], p['relation']['class'], p['object']['class']) for p in predictions]

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

        TP = len(predicted_set.intersection(ground_truth_set))  # True Positives
        FP = len(predicted_set - ground_truth_set)  # False Positives
        FN = len(ground_truth_set - predicted_set)  # False Negatives
        
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

    def calculate_roc_metrics(self, predictions, min_pairs_range=range(1, 6)):
        """
        Tính toán ROC curve cho các ngưỡng min_pairs khác nhau.
        """
        results = {
            'pairs': [],
            'triplets': []
        }
        
        for min_pairs in min_pairs_range:
            # Đánh giá dựa trên cặp subject-object
            pairs_results = self.query_images_by_pairs_count(predictions, min_pairs)
            pairs_metrics = self._calculate_metrics(pairs_results)
            results['pairs'].append({
                'min_pairs': min_pairs,
                'metrics': pairs_metrics
            })
            
            # Đánh giá dựa trên triplets
            triplets_results = self.query_images_by_full_pairs_count(predictions, min_pairs)
            triplets_metrics = self._calculate_metrics(triplets_results)
            results['triplets'].append({
                'min_pairs': min_pairs,
                'metrics': triplets_metrics
            })
        
        return results

    def _calculate_metrics(self, results):
        """
        Tính toán các metrics từ kết quả truy vấn.
        """
        if not results:
            return {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'matching_percentage': 0
            }
        
        total_matching = sum(r['matching_pairs'] if 'matching_pairs' in r else r['matching_triples'] for r in results)
        total_pairs = sum(r['total_pairs'] if 'total_pairs' in r else r['total_triples'] for r in results)
        total_images = len(results)
        
        precision = total_matching / total_pairs if total_pairs > 0 else 0
        recall = total_matching / (total_pairs * total_images) if total_pairs * total_images > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        matching_percentage = sum(r['matching_percentage'] for r in results) / total_images if total_images > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'matching_percentage': matching_percentage
        }

    def plot_roc_curves(self, results, output_file='roc_curves.png'):
        """
        Vẽ ROC curves cho các kết quả đánh giá theo từng danh mục.
        """
        plt.figure(figsize=(15, 10))
        
        # Tạo subplot cho ROC curves
        plt.subplot(2, 2, 1)
        plt.title('ROC Curves for Subject-Object Pairs')
        
        # Vẽ ROC curve cho cặp subject-object
        for result in results['pairs']:
            min_pairs = result['min_pairs']
            metrics = result['metrics']
            plt.plot(metrics['recall'], metrics['precision'], 
                    label=f'Min {min_pairs} pairs (F1={metrics["f1_score"]:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        
        # Tạo subplot cho ROC curves của triplets
        plt.subplot(2, 2, 2)
        plt.title('ROC Curves for Triplets')
        
        # Vẽ ROC curve cho triplets
        for result in results['triplets']:
            min_pairs = result['min_pairs']
            metrics = result['metrics']
            plt.plot(metrics['recall'], metrics['precision'], 
                    label=f'Min {min_pairs} triplets (F1={metrics["f1_score"]:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        
        # Tạo subplot cho Precision-Recall của cặp subject-object
        plt.subplot(2, 2, 3)
        plt.title('Precision-Recall for Subject-Object Pairs')
        
        for result in results['pairs']:
            min_pairs = result['min_pairs']
            metrics = result['metrics']
            plt.plot([0, 1], [metrics['precision'], metrics['precision']], 
                    label=f'Min {min_pairs} pairs (P={metrics["precision"]:.2f})')
            plt.plot([0, 1], [metrics['recall'], metrics['recall']], 
                    label=f'Min {min_pairs} pairs (R={metrics["recall"]:.2f})')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        # Tạo subplot cho Precision-Recall của triplets
        plt.subplot(2, 2, 4)
        plt.title('Precision-Recall for Triplets')
        
        for result in results['triplets']:
            min_pairs = result['min_pairs']
            metrics = result['metrics']
            plt.plot([0, 1], [metrics['precision'], metrics['precision']], 
                    label=f'Min {min_pairs} triplets (P={metrics["precision"]:.2f})')
            plt.plot([0, 1], [metrics['recall'], metrics['recall']], 
                    label=f'Min {min_pairs} triplets (R={metrics["recall"]:.2f})')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"ROC curves and Precision-Recall saved to {output_file}")

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
            predictions = predict(image_path, self.model)
            return image_id, predictions
        except Exception as e:
            logger.error(f"Lỗi khi xử lý ảnh {image_id}: {str(e)}")
            return image_id, None

    def _batch_query_neo4j(self, predictions_batch):
        """
        Truy vấn Neo4j cho một batch dự đoán.
        """
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
            
            # Thực hiện truy vấn và lưu kết quả
            pairs_result = list(session.run(pairs_query, params))
            triplets_result = list(session.run(triplets_query, params))
            
            return pairs_result, triplets_result

    def evaluate_all_images(self, batch_size=None, max_images=None, specific_images=None):
        """
        Đánh giá các ảnh trong thư mục.
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
        
        all_results = {
            'pairs': [],
            'triplets': []
        }
        
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
                
                if batch_predictions:
                    # Truy vấn Neo4j cho batch
                    pairs_result, triplets_result = self._batch_query_neo4j(batch_predictions)
                    
                    # Cập nhật kết quả
                    self._update_results(all_results, pairs_result, triplets_result)
                
                # Giải phóng bộ nhớ sau mỗi batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        return all_results

    def _update_results(self, all_results, pairs_result, triplets_result):
        """
        Cập nhật kết quả từ truy vấn Neo4j.
        """
        # Cập nhật kết quả cho cặp subject-object
        for record in pairs_result:
            matching_pairs = record['matching_pairs']
            # Cập nhật cho tất cả các ngưỡng từ 1 đến 5
            for min_pairs in range(1, 6):
                if matching_pairs >= min_pairs:
                    self._update_metrics(all_results['pairs'], record, 'matching_pairs', 'total_pairs', min_pairs)
        
        # Cập nhật kết quả cho triplets
        for record in triplets_result:
            matching_triples = record['matching_triples']
            # Cập nhật cho tất cả các ngưỡng từ 1 đến 5
            for min_pairs in range(1, 6):
                if matching_triples >= min_pairs:
                    self._update_metrics(all_results['triplets'], record, 'matching_triples', 'total_triples', min_pairs)

    def _update_metrics(self, results, record, matching_field, total_field, min_pairs):
        """
        Cập nhật metrics cho một kết quả.
        """
        # Tìm hoặc tạo kết quả cho ngưỡng min_pairs
        while len(results) < min_pairs:
            results.append({
                'min_pairs': len(results) + 1,
                'metrics': {
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0,
                    'matching_percentage': 0,
                    'total_images': 0
                }
            })
        
        current_metrics = results[min_pairs - 1]['metrics']
        matching = record[matching_field]
        total = record[total_field]
        
        # Cập nhật metrics
        total_images = current_metrics['total_images']
        new_total = total_images + 1
        
        precision = matching / total if total > 0 else 0
        recall = matching / (total * new_total) if total * new_total > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        current_metrics['precision'] = (current_metrics['precision'] * total_images + precision) / new_total
        current_metrics['recall'] = (current_metrics['recall'] * total_images + recall) / new_total
        current_metrics['f1_score'] = (current_metrics['f1_score'] * total_images + f1_score) / new_total
        current_metrics['matching_percentage'] = (current_metrics['matching_percentage'] * total_images + record['matching_percentage']) / new_total
        current_metrics['total_images'] = new_total

def main():
    """Hàm chính để thực hiện đánh giá"""
    try:
        evaluator = RelTREvaluator()
        
        # Đánh giá số lượng ảnh cụ thể
        results = evaluator.evaluate_all_images(max_images=100)
        
        # Vẽ ROC curves và Precision-Recall
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"roc_curves_{timestamp}.png"
        evaluator.plot_roc_curves(results, output_file)
        
        # Lưu kết quả
        results_file = f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Kết quả đánh giá đã được lưu vào {results_file}")
        logger.info(f"Biểu đồ ROC và Precision-Recall đã được lưu vào {output_file}")
        
        # In thống kê tổng quan
        logger.info("\nThống kê tổng quan:")
        for metric_type in ['pairs', 'triplets']:
            logger.info(f"\nKết quả cho {metric_type}:")
            for result in results[metric_type]:
                metrics = result['metrics']
                logger.info(f"Min {result['min_pairs']} {metric_type}:")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall: {metrics['recall']:.4f}")
                logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
                logger.info(f"  Matching Percentage: {metrics['matching_percentage']:.2f}%")
                logger.info(f"  Total Images: {metrics['total_images']}")
        
    except Exception as e:
        logger.error(f"Lỗi trong hàm main: {str(e)}")

if __name__ == "__main__":
    main()