import os
import json
import logging
import time
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from neo4j import GraphDatabase
from RelTR.inference import load_model, predict
import shutil
import glob
import sys
import argparse
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Cấu hình thư mục
IMAGE_FOLDER = './data/vg_focused/images'  # Thư mục chứa hình ảnh
UPLOAD_FOLDER = './uploads'  # Thư mục để lưu ảnh upload
OUTPUT_FOLDER = './output_images'  # Thư mục để lưu ảnh output
RESULTS_FOLDER = './query_results'  # Thư mục để lưu kết quả truy vấn
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Kết nối Neo4j
uri = "bolt://localhost:7689"
# uri = "neo4j+s://b40b4f2a.databases.neo4j.io"
username = "neo4j"
# password = "fpKNUXKT-4z0kQMm1nuUaiXe8p70uIebc3y3a4Z8kUA"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Load mô hình RelTR
try:
    logger.info("Đang load mô hình RelTR...")
    model = load_model('./RelTR/ckpt/fine_tune1/checkpoint0049.pth')
    logger.info("Đã load mô hình RelTR thành công")
except Exception as e:
    logger.error(f"Lỗi khi load mô hình RelTR: {str(e)}")
    raise

def query_images_by_pairs_count(predictions, min_pairs):
    image_details = []
    processed_image_ids = set()
    query_start_time = time.time()
    
    with driver.session() as session:
        subjects = set(p['subject']['class'] for p in predictions)
        objects = set(p['object']['class'] for p in predictions)
        prediction_pairs = [(p['subject']['class'], p['object']['class']) for p in predictions]
        
        # Query chính để tìm ảnh và đếm cặp khớp
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
        
        total_results = 0
        total_matching_percentage = 0
        
        for record in result:
            image_id = record['image_id']
            
            # Kiểm tra sự tồn tại của file ảnh
            image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{image_id}.jpg")
            if not os.path.exists(image_path):
                continue
                
            if image_id not in processed_image_ids:
                processed_image_ids.add(image_id)
                relationships_str = []
                for rel in record['relationships']:
                    relationships_str.append(
                        f"{rel['subject']} -[{rel['relation']}]-> {rel['object']}"
                    )
                total_matching_percentage += record['matching_percentage']
                
                image_details.append({
                    "image_id": image_id,
                    "matching_pairs": record['matching_pairs'],
                    "total_pairs": record['total_pairs'],
                    "matching_percentage": round(record['matching_percentage'], 2),
                    "relationships": relationships_str
                })
                total_results += 1
    
    query_time = time.time() - query_start_time
    avg_matching_percentage = total_matching_percentage / total_results if total_results > 0 else 0
    
    return image_details

def query_images_by_full_pairs_count(predictions, min_pairs):
    image_details = []
    processed_image_ids = set()
    query_start_time = time.time()
    
    with driver.session() as session:
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
        
        total_results = 0
        total_matching_percentage = 0
        matching_triples_distribution = {}
        
        for record in result:
            total_results += 1
            image_id = record['image_id']
            matching_triples = record['matching_triples']
            
            matching_triples_distribution[matching_triples] = matching_triples_distribution.get(matching_triples, 0) + 1
            
            if image_id not in processed_image_ids:
                processed_image_ids.add(image_id)
                relationships_str = []
                for rel in record['relationships']:
                    relationships_str.append(
                        f"{rel['subject']} -[{rel['relation']}]-> {rel['object']}"
                    )
                total_matching_percentage += record['matching_percentage']
                
                image_details.append({
                    "image_id": image_id,
                    "matching_triples": matching_triples,
                    "total_triples": record['total_triples'],
                    "matching_percentage": round(record['matching_percentage'], 2),
                    "relationships": relationships_str
                })
    
    query_time = time.time() - query_start_time
    avg_matching_percentage = total_matching_percentage / total_results if total_results > 0 else 0
    
    return image_details

def find_images_with_matching_pairs(predictions):
    results = {}
    
    with driver.session() as session:
        subjects = set(p['subject']['class'] for p in predictions)
        subjects_query = """
        MATCH (s:Object)
        WHERE s.category IN $subjects
        RETURN DISTINCT s.image_id as image_id
        """
        subjects_result = session.run(subjects_query, {'subjects': list(subjects)})
        subject_images = set(record['image_id'] for record in subjects_result)
        
        objects = set(p['object']['class'] for p in predictions)
        objects_query = """
        MATCH (s:Object)
        WHERE s.category IN $objects
        RETURN DISTINCT s.image_id as image_id
        """
        objects_result = session.run(objects_query, {'objects': list(objects)})
        object_images = set(record['image_id'] for record in objects_result)
        
        common_images = subject_images.intersection(object_images)
        
        for pred in predictions:
            subject = pred['subject']['class']
            object = pred['object']['class']
            relation = pred['relation']['class']
            
            query = """
            MATCH (s:Object)
            WHERE s.category = $subject
            WITH DISTINCT s.image_id as image_id
            MATCH (o:Object)
            WHERE o.image_id = image_id AND o.category = $object
            WITH image_id
            MATCH (s2:Object)-[r:RELATIONSHIP]->(o2:Object)
            WHERE s2.image_id = image_id
            WITH image_id,
                 COLLECT(DISTINCT {subject: s2.category, relation: r.type, object: o2.category}) as relationships
            RETURN image_id, relationships
            ORDER BY image_id
            """
            
            result = session.run(query, {'subject': subject, 'object': object})
            
            pair_key = f"{subject}-{relation}-{object}"
            results[pair_key] = []
            
            for record in result:
                image_id = record['image_id']
                relationships = []
                for rel in record['relationships']:
                    relationships.append(
                        f"{rel['subject']} -[{rel['relation']}]-> {rel['object']}"
                    )
                
                results[pair_key].append({
                    "image_id": image_id,
                    "url": f"data/vg_focused/images/{image_id}.jpg",
                    "relationships": relationships
                })
    
    return results

def calculate_metrics(predictions, query_results):
    """
    Tính toán các metrics cần thiết cho ROC curve và Precision-Recall curve
    dựa trên kết quả truy vấn và tổng số ảnh được truy vấn
    """
    metrics = {}
    
    # Tính tổng số ảnh duy nhất được truy vấn
    all_retrieved_images = set()
    for threshold in range(1, 6):
        key = f"{threshold}_or_more"
        if key in query_results:
            for result in query_results[key]:
                all_retrieved_images.add(result['image_id'])
        key_full = f"{threshold}_or_more_full"
        if key_full in query_results:
            for result in query_results[key_full]:
                all_retrieved_images.add(result['image_id'])
    
    total_retrieved = len(all_retrieved_images)
    logger.info(f"Tổng số ảnh duy nhất được truy vấn: {total_retrieved}")
    
    # Xử lý kết quả truy vấn theo cặp
    for threshold in range(1, 6):
        key = f"{threshold}_or_more"
        if key in query_results:
            results = query_results[key]
            
            # Đếm số lượng ảnh duy nhất được truy vấn
            unique_images = set()
            for result in results:
                unique_images.add(result['image_id'])
            
            # Tính toán các giá trị
            retrieved_count = len(unique_images)
            
            # Giả sử tất cả ảnh được truy vấn là true positives
            tp = retrieved_count
            fp = 0  # Không có false positives vì không có ground truth
            fn = 0  # Không có false negatives vì không có ground truth
            tn = total_retrieved - retrieved_count  # Các ảnh còn lại là true negatives
            
            # Tính precision, recall và F1-score với xử lý chia cho 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Thêm log để debug
            logger.info(f"Metrics for {key}:")
            logger.info(f"Retrieved count: {retrieved_count}")
            logger.info(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
            
            metrics[key] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
    
    # Xử lý kết quả truy vấn theo bộ ba
    for threshold in range(1, 6):
        key = f"{threshold}_or_more_full"
        if key in query_results:
            results = query_results[key]
            
            # Đếm số lượng ảnh duy nhất được truy vấn
            unique_images = set()
            for result in results:
                unique_images.add(result['image_id'])
            
            # Tính toán các giá trị
            retrieved_count = len(unique_images)
            
            # Giả sử tất cả ảnh được truy vấn là true positives
            tp = retrieved_count
            fp = 0
            fn = 0
            tn = total_retrieved - retrieved_count
            
            # Tính precision, recall và F1-score với xử lý chia cho 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Thêm log để debug
            logger.info(f"Metrics for {key}:")
            logger.info(f"Retrieved count: {retrieved_count}")
            logger.info(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
            
            metrics[key] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
    
    return metrics

def plot_roc_curve(metrics, output_path):
    """
    Vẽ ROC curve từ metrics đã tính toán
    """
    try:
        plt.figure(figsize=(10, 6))
        
        for method, values in metrics.items():
            # Tính TPR (True Positive Rate) và FPR (False Positive Rate) với xử lý chia cho 0
            denominator_tpr = values['tp'] + values['fn']
            denominator_fpr = values['fp'] + values['tn']
            
            tpr = values['tp'] / denominator_tpr if denominator_tpr > 0 else 0.0
            fpr = values['fp'] / denominator_fpr if denominator_fpr > 0 else 0.0
            
            # Log để debug
            logger.info(f"ROC metrics for {method}:")
            logger.info(f"TPR: {tpr:.4f}, FPR: {fpr:.4f}")
            logger.info(f"TP: {values['tp']}, FN: {values['fn']}, FP: {values['fp']}, TN: {values['tn']}")
            
            plt.plot(fpr, tpr, marker='o', label=method)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
    except Exception as e:
        logger.error(f"Lỗi khi vẽ ROC curve: {str(e)}")
        raise

def plot_precision_recall_curve(metrics, output_path):
    """
    Vẽ Precision-Recall curve từ metrics đã tính toán
    """
    try:
        plt.figure(figsize=(10, 6))
        
        for method, values in metrics.items():
            precision = values['precision']
            recall = values['recall']
            
            # Log để debug
            logger.info(f"PR metrics for {method}:")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            plt.plot(recall, precision, marker='o', label=method)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
    except Exception as e:
        logger.error(f"Lỗi khi vẽ Precision-Recall curve: {str(e)}")
        raise

def process_image(image_path):
    """
    Xử lý một ảnh và trả về metrics
    """
    try:
        # Dự đoán các mối quan hệ trong ảnh
        predictions = predict(image_path, model)
        
        # Truy vấn ảnh với các ngưỡng khác nhau
        results = {}
        for min_pairs in range(1, 6):
            key = f"{min_pairs}_or_more"
            results[key] = query_images_by_pairs_count(predictions, min_pairs)
            
            key_full = f"{min_pairs}_or_more_full"
            results[key_full] = query_images_by_full_pairs_count(predictions, min_pairs)
        
        # Tính toán metrics
        metrics = calculate_metrics(predictions, results)
        
        # Ghi log
        with open("log.txt", "a") as f:
            f.write(f"\n=== Evaluation for {image_path} ===\n")
            for method, values in metrics.items():
                f.write(f"\n{method}:\n")
                f.write(f"Precision: {values['precision']:.4f}\n")
                f.write(f"Recall: {values['recall']:.4f}\n")
                f.write(f"F1-score: {values['f1_score']:.4f}\n")
            f.write("\n")
            
        return metrics
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
        return None

def aggregate_metrics(all_metrics):
    """
    Tổng hợp metrics từ tất cả các ảnh
    """
    aggregated = {}
    
    # Khởi tạo cấu trúc dữ liệu cho metrics tổng hợp
    for threshold in range(1, 6):
        for suffix in ['_or_more', '_or_more_full']:
            key = f"{threshold}{suffix}"
            aggregated[key] = {
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'count': 0  # Số lượng ảnh có metrics hợp lệ
            }
    
    # Tổng hợp metrics từ tất cả các ảnh
    for metrics in all_metrics:
        if metrics is None:
            continue
            
        for key, values in metrics.items():
            if key in aggregated:
                aggregated[key]['tp'] += values['tp']
                aggregated[key]['fp'] += values['fp']
                aggregated[key]['fn'] += values['fn']
                aggregated[key]['tn'] += values['tn']
                aggregated[key]['count'] += 1
    
    # Tính trung bình cho mỗi metric
    for key in aggregated:
        count = aggregated[key]['count']
        if count > 0:
            aggregated[key]['tp'] //= count
            aggregated[key]['fp'] //= count
            aggregated[key]['fn'] //= count
            aggregated[key]['tn'] //= count
            
            # Tính lại precision, recall và F1-score
            tp = aggregated[key]['tp']
            fp = aggregated[key]['fp']
            fn = aggregated[key]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            aggregated[key]['precision'] = precision
            aggregated[key]['recall'] = recall
            aggregated[key]['f1_score'] = f1_score
            
            # Log metrics tổng hợp
            logger.info(f"\nAggregated metrics for {key}:")
            logger.info(f"Average TP: {tp}, FP: {fp}, FN: {fn}, TN: {aggregated[key]['tn']}")
            logger.info(f"Average Precision: {precision:.4f}")
            logger.info(f"Average Recall: {recall:.4f}")
            logger.info(f"Average F1-score: {f1_score:.4f}")
            logger.info(f"Number of images: {count}")
    
    return aggregated

def process_directory(directory_path):
    """
    Xử lý tất cả ảnh trong thư mục
    """
    try:
        # Tạo thư mục plots nếu chưa tồn tại
        os.makedirs("plots", exist_ok=True)
        
        # Lấy danh sách tất cả các file ảnh
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(directory_path, f"*{ext.upper()}")))
        
        if not image_files:
            logger.warning(f"Không tìm thấy file ảnh nào trong thư mục {directory_path}")
            return False
            
        # Xử lý từng ảnh và thu thập metrics
        all_metrics = []
        for image_path in image_files:
            logger.info(f"Đang xử lý ảnh: {image_path}")
            metrics = process_image(image_path)
            if metrics:
                all_metrics.append(metrics)
                
        if not all_metrics:
            logger.warning("Không có metrics nào được thu thập")
            return False
            
        # Tổng hợp metrics và vẽ biểu đồ
        aggregated_metrics = aggregate_metrics(all_metrics)
        
        # Vẽ biểu đồ tổng hợp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        roc_path = f"plots/roc_aggregated_{timestamp}.png"
        pr_path = f"plots/pr_aggregated_{timestamp}.png"
        
        plot_roc_curve(aggregated_metrics, roc_path)
        plot_precision_recall_curve(aggregated_metrics, pr_path)
        
        # Lưu metrics tổng hợp vào summary.json
        summary = {
            "timestamp": timestamp,
            "total_images": len(all_metrics),
            "metrics": aggregated_metrics,
            "roc_plot": roc_path,
            "pr_plot": pr_path
        }
        
        with open("summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Đã xử lý thành công {len(all_metrics)}/{len(image_files)} ảnh")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý thư mục {directory_path}: {str(e)}")
        return False

def main():
    """
    Hàm chính để chạy script
    """
    try:
        parser = argparse.ArgumentParser(description="Tự động truy vấn dữ liệu ảnh và lưu kết quả")
        parser.add_argument("--directory", help="Đường dẫn đến thư mục chứa ảnh cần xử lý")
        parser.add_argument("input_dir", nargs="?", help="Đường dẫn đến thư mục chứa ảnh cần xử lý")
        
        args = parser.parse_args()
        
        # Xác định thư mục đầu vào
        input_directory = args.directory if args.directory else args.input_dir
        
        if not input_directory:
            print("Usage: python auto_query.py <input_directory>")
            print("   or: python auto_query.py --directory <input_directory>")
            return
            
        if not os.path.exists(input_directory):
            print(f"Thư mục {input_directory} không tồn tại")
            return
            
        # Xử lý thư mục đầu vào
        if process_directory(input_directory):
            print("Xử lý thành công")
        else:
            print("Xử lý thất bại")
            
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        
if __name__ == "__main__":
    main() 