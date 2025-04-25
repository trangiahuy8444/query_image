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
uri = "neo4j+s://b40b4f2a.databases.neo4j.io"
username = "neo4j"
password = "fpKNUXKT-4z0kQMm1nuUaiXe8p70uIebc3y3a4Z8kUA"
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

def calculate_metrics(predictions, query_results, total_images_in_db):
    """
    Tính toán các số liệu để vẽ biểu đồ ROC và precision-recall
    
    Args:
        predictions: Danh sách các dự đoán từ mô hình
        query_results: Kết quả truy vấn từ các phương pháp khác nhau
        total_images_in_db: Tổng số ảnh trong cơ sở dữ liệu
    
    Returns:
        Dictionary chứa các số liệu cho ROC và precision-recall
    """
    metrics = {}
    
    # Xử lý kết quả từ query_images_by_pairs_count
    for threshold, results in query_results.get('related_images', {}).items():
        tp = len(results)  # True positives - ảnh được truy xuất đúng
        fp = 0  # False positives - cần ground truth để tính
        fn = 0  # False negatives - cần ground truth để tính
        tn = total_images_in_db - tp  # True negatives - ảnh còn lại
        
        # Tính các chỉ số cơ bản
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'pairs_{threshold}'] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_retrieved': len(results)
        }
    
    # Xử lý kết quả từ query_images_by_full_pairs_count
    for threshold, results in query_results.get('related_images_full', {}).items():
        tp = len(results)
        fp = 0
        fn = 0
        tn = total_images_in_db - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'full_triples_{threshold}'] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_retrieved': len(results)
        }
    
    # Xử lý kết quả từ find_images_with_matching_pairs
    matching_pairs_results = query_results.get('matching_pairs_results', {})
    total_matching_pairs = sum(len(images) for images in matching_pairs_results.values())
    
    metrics['matching_pairs'] = {
        'total_retrieved': total_matching_pairs,
        'unique_images': len(set(img['image_id'] for images in matching_pairs_results.values() for img in images))
    }
    
    return metrics

def plot_roc_curve(metrics, output_path):
    """
    Vẽ biểu đồ ROC curve từ các số liệu đã tính
    
    Args:
        metrics: Dictionary chứa các số liệu
        output_path: Đường dẫn để lưu biểu đồ
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(10, 8))
        
        # Vẽ ROC curve cho các phương pháp truy vấn theo cặp
        for threshold in ['1_or_more', '2_or_more', '3_or_more', '4_or_more', '5_or_more']:
            if f'pairs_{threshold}' in metrics:
                m = metrics[f'pairs_{threshold}']
                tpr = m['recall']  # True Positive Rate = Recall
                fpr = m['false_positives'] / (m['false_positives'] + m['true_negatives']) if (m['false_positives'] + m['true_negatives']) > 0 else 0
                plt.plot(fpr, tpr, 'o-', label=f'Pairs {threshold}')
        
        # Vẽ ROC curve cho các phương pháp truy vấn theo bộ ba
        for threshold in ['1_or_more_full', '2_or_more_full', '3_or_more_full', '4_or_more_full', '5_or_more_full']:
            if f'full_triples_{threshold}' in metrics:
                m = metrics[f'full_triples_{threshold}']
                tpr = m['recall']
                fpr = m['false_positives'] / (m['false_positives'] + m['true_negatives']) if (m['false_positives'] + m['true_negatives']) > 0 else 0
                plt.plot(fpr, tpr, 's-', label=f'Full Triples {threshold}')
        
        # Vẽ đường chéo ngẫu nhiên
        plt.plot([0, 1], [0, 1], 'k--')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ ROC: {str(e)}")
        return False

def plot_precision_recall_curve(metrics, output_path):
    """
    Vẽ biểu đồ Precision-Recall curve từ các số liệu đã tính
    
    Args:
        metrics: Dictionary chứa các số liệu
        output_path: Đường dẫn để lưu biểu đồ
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(10, 8))
        
        # Vẽ Precision-Recall curve cho các phương pháp truy vấn theo cặp
        for threshold in ['1_or_more', '2_or_more', '3_or_more', '4_or_more', '5_or_more']:
            if f'pairs_{threshold}' in metrics:
                m = metrics[f'pairs_{threshold}']
                precision = m['precision']
                recall = m['recall']
                plt.plot(recall, precision, 'o-', label=f'Pairs {threshold}')
        
        # Vẽ Precision-Recall curve cho các phương pháp truy vấn theo bộ ba
        for threshold in ['1_or_more_full', '2_or_more_full', '3_or_more_full', '4_or_more_full', '5_or_more_full']:
            if f'full_triples_{threshold}' in metrics:
                m = metrics[f'full_triples_{threshold}']
                precision = m['precision']
                recall = m['recall']
                plt.plot(recall, precision, 's-', label=f'Full Triples {threshold}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ Precision-Recall: {str(e)}")
        return False

def process_image(image_path):
    """
    Xử lý một ảnh, thực hiện dự đoán và truy vấn, sau đó lưu kết quả
    """
    try:
        # Tạo thư mục kết quả với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_folder = os.path.join(app.config['RESULTS_FOLDER'], timestamp)
        os.makedirs(result_folder, exist_ok=True)
        
        # Lưu ảnh gốc vào thư mục kết quả
        filename = os.path.basename(image_path)
        shutil.copy2(image_path, os.path.join(result_folder, filename))
        
        # Thực hiện dự đoán
        prediction_start_time = time.time()
        predictions = predict(image_path, model)
        prediction_time = time.time() - prediction_start_time
        
        # Lưu kết quả dự đoán
        with open(os.path.join(result_folder, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=2)
        
        # Truy vấn ảnh theo cặp
        related_images = {
            "1_or_more": query_images_by_pairs_count(predictions, 1),
            "2_or_more": query_images_by_pairs_count(predictions, 2),
            "3_or_more": query_images_by_pairs_count(predictions, 3),
            "4_or_more": query_images_by_pairs_count(predictions, 4),
            "5_or_more": query_images_by_pairs_count(predictions, 5),
        }
        
        # Lưu kết quả truy vấn theo cặp
        with open(os.path.join(result_folder, "related_images.json"), "w") as f:
            json.dump(related_images, f, indent=2)
        
        # Truy vấn ảnh theo bộ ba
        related_images_full = {
            "1_or_more_full": query_images_by_full_pairs_count(predictions, 1),
            "2_or_more_full": query_images_by_full_pairs_count(predictions, 2),
            "3_or_more_full": query_images_by_full_pairs_count(predictions, 3),
            "4_or_more_full": query_images_by_full_pairs_count(predictions, 4),
            "5_or_more_full": query_images_by_full_pairs_count(predictions, 5),
        }
        
        # Lưu kết quả truy vấn theo bộ ba
        with open(os.path.join(result_folder, "related_images_full.json"), "w") as f:
            json.dump(related_images_full, f, indent=2)
        
        # Truy vấn ảnh theo matching pairs
        matching_pairs_results = find_images_with_matching_pairs(predictions)
        
        # Lưu kết quả matching pairs
        with open(os.path.join(result_folder, "matching_pairs_results.json"), "w") as f:
            json.dump(matching_pairs_results, f, indent=2)
        
        # Tạo thư mục output cho ảnh
        output_folder = os.path.join(app.config['OUTPUT_FOLDER'], os.path.splitext(filename)[0])
        os.makedirs(output_folder, exist_ok=True)
        
        # Lưu các ảnh liên quan
        saved_images = set()
        
        for category, images in related_images.items():
            for image in images:
                if image['image_id'] not in saved_images:
                    image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{image['image_id']}.jpg")
                    if os.path.exists(image_path):
                        shutil.copy2(image_path, os.path.join(output_folder, f"{image['image_id']}.jpg"))
                        saved_images.add(image['image_id'])
        
        for category, images in related_images_full.items():
            for image in images:
                if image['image_id'] not in saved_images:
                    image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{image['image_id']}.jpg")
                    if os.path.exists(image_path):
                        shutil.copy2(image_path, os.path.join(output_folder, f"{image['image_id']}.jpg"))
                        saved_images.add(image['image_id'])
        
        # Đếm số lượng ảnh thực tế có thể hiển thị
        counted_images = set()
        total_images = 0
        
        for category in related_images:
            for image in related_images[category]:
                image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{image['image_id']}.jpg")
                if os.path.exists(image_path):
                    image["url"] = f"data/vg_focused/images/{image['image_id']}.jpg"
                    
                    if image['image_id'] not in counted_images:
                        counted_images.add(image['image_id'])
                        total_images += 1

        for category in related_images_full:
            for image in related_images_full[category]:
                image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{image['image_id']}.jpg")
                if os.path.exists(image_path):
                    image["url"] = f"data/vg_focused/images/{image['image_id']}.jpg"
                    
                    if image['image_id'] not in counted_images:
                        counted_images.add(image['image_id'])
                        total_images += 1
        
        # Tính toán các số liệu để vẽ biểu đồ ROC và precision-recall
        total_images_in_db = 108077  # Số ảnh trong cơ sở dữ liệu (cần cập nhật nếu thay đổi)
        query_results = {
            'related_images': related_images,
            'related_images_full': related_images_full,
            'matching_pairs_results': matching_pairs_results
        }
        metrics = calculate_metrics(predictions, query_results, total_images_in_db)
        
        # Vẽ biểu đồ ROC và precision-recall
        plot_roc_curve(metrics, os.path.join(result_folder, "roc_curve.png"))
        plot_precision_recall_curve(metrics, os.path.join(result_folder, "precision_recall_curve.png"))
        
        # Lưu thông tin tổng hợp
        summary = {
            "image_path": image_path,
            "prediction_time": round(prediction_time, 2),
            "num_predictions": len(predictions),
            "total_unique_images": total_images,
            "related_images_counts": {k: len(v) for k, v in related_images.items()},
            "related_images_full_counts": {k: len(v) for k, v in related_images_full.items()},
            "matching_pairs_counts": {k: len(v) for k, v in matching_pairs_results.items()},
            "metrics": metrics,
            "timestamp": timestamp
        }
        
        with open(os.path.join(result_folder, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        # Lưu log
        with open(os.path.join(result_folder, "log.txt"), "w") as f:
            f.write(f"Xử lý ảnh: {image_path}\n")
            f.write(f"Thời gian dự đoán: {prediction_time:.2f} giây\n")
            f.write(f"Số lượng dự đoán: {len(predictions)}\n")
            f.write(f"Tổng số ảnh duy nhất được truy vấn: {total_images}\n")
            f.write(f"Kết quả truy vấn theo cặp: {json.dumps({k: len(v) for k, v in related_images.items()}, indent=2)}\n")
            f.write(f"Kết quả truy vấn theo bộ ba: {json.dumps({k: len(v) for k, v in related_images_full.items()}, indent=2)}\n")
            f.write(f"Kết quả matching pairs: {json.dumps({k: len(v) for k, v in matching_pairs_results.items()}, indent=2)}\n")
            f.write(f"Kết quả đánh giá mô hình:\n")
            for method, m in metrics.items():
                if isinstance(m, dict) and 'precision' in m and 'recall' in m:
                    f.write(f"  {method}:\n")
                    f.write(f"    Precision: {m['precision']:.4f}\n")
                    f.write(f"    Recall: {m['recall']:.4f}\n")
                    f.write(f"    F1-score: {m['f1_score']:.4f}\n")
        
        logger.info(f"Đã xử lý ảnh {image_path} và lưu kết quả vào {result_folder}")
        return result_folder
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
        return None

def process_directory(directory_path):
    """
    Xử lý tất cả ảnh trong một thư mục
    """
    results = []
    all_metrics = {
        'pairs': {},
        'full_triples': {},
        'matching_pairs': {'total_retrieved': 0, 'unique_images': 0}
    }
    
    # Lưu trữ metrics của từng ảnh riêng biệt
    individual_metrics = {
        'pairs': {},
        'full_triples': {}
    }
    
    # Tạo thư mục kết quả tổng hợp với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate_folder = os.path.join(app.config['RESULTS_FOLDER'], f"aggregate_{timestamp}")
    os.makedirs(aggregate_folder, exist_ok=True)
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            result_folder = process_image(image_path)
            if result_folder:
                results.append(result_folder)
                
                # Đọc metrics từ file summary.json
                try:
                    with open(os.path.join(result_folder, "summary.json"), "r") as f:
                        summary = json.load(f)
                        if 'metrics' in summary:
                            metrics = summary['metrics']
                            
                            # Tổng hợp metrics cho pairs
                            for key, value in metrics.items():
                                if key.startswith('pairs_'):
                                    threshold = key.replace('pairs_', '')
                                    if threshold not in all_metrics['pairs']:
                                        all_metrics['pairs'][threshold] = {
                                            'true_positives': 0,
                                            'false_positives': 0,
                                            'false_negatives': 0,
                                            'true_negatives': 0,
                                            'precision': 0,
                                            'recall': 0,
                                            'f1_score': 0,
                                            'total_retrieved': 0,
                                            'count': 0
                                        }
                                    
                                    # Lưu metrics của ảnh hiện tại
                                    if threshold not in individual_metrics['pairs']:
                                        individual_metrics['pairs'][threshold] = []
                                    
                                    individual_metrics['pairs'][threshold].append({
                                        'image_id': os.path.basename(image_path),
                                        'true_positives': value['true_positives'],
                                        'false_positives': value['false_positives'],
                                        'false_negatives': value['false_negatives'],
                                        'true_negatives': value['true_negatives'],
                                        'precision': value['precision'],
                                        'recall': value['recall'],
                                        'f1_score': value['f1_score'],
                                        'total_retrieved': value['total_retrieved']
                                    })
                                    
                                    all_metrics['pairs'][threshold]['true_positives'] += value['true_positives']
                                    all_metrics['pairs'][threshold]['false_positives'] += value['false_positives']
                                    all_metrics['pairs'][threshold]['false_negatives'] += value['false_negatives']
                                    all_metrics['pairs'][threshold]['true_negatives'] += value['true_negatives']
                                    all_metrics['pairs'][threshold]['total_retrieved'] += value['total_retrieved']
                                    all_metrics['pairs'][threshold]['count'] += 1
                            
                            # Tổng hợp metrics cho full_triples
                            for key, value in metrics.items():
                                if key.startswith('full_triples_'):
                                    threshold = key.replace('full_triples_', '')
                                    if threshold not in all_metrics['full_triples']:
                                        all_metrics['full_triples'][threshold] = {
                                            'true_positives': 0,
                                            'false_positives': 0,
                                            'false_negatives': 0,
                                            'true_negatives': 0,
                                            'precision': 0,
                                            'recall': 0,
                                            'f1_score': 0,
                                            'total_retrieved': 0,
                                            'count': 0
                                        }
                                    
                                    # Lưu metrics của ảnh hiện tại
                                    if threshold not in individual_metrics['full_triples']:
                                        individual_metrics['full_triples'][threshold] = []
                                    
                                    individual_metrics['full_triples'][threshold].append({
                                        'image_id': os.path.basename(image_path),
                                        'true_positives': value['true_positives'],
                                        'false_positives': value['false_positives'],
                                        'false_negatives': value['false_negatives'],
                                        'true_negatives': value['true_negatives'],
                                        'precision': value['precision'],
                                        'recall': value['recall'],
                                        'f1_score': value['f1_score'],
                                        'total_retrieved': value['total_retrieved']
                                    })
                                    
                                    all_metrics['full_triples'][threshold]['true_positives'] += value['true_positives']
                                    all_metrics['full_triples'][threshold]['false_positives'] += value['false_positives']
                                    all_metrics['full_triples'][threshold]['false_negatives'] += value['false_negatives']
                                    all_metrics['full_triples'][threshold]['true_negatives'] += value['true_negatives']
                                    all_metrics['full_triples'][threshold]['total_retrieved'] += value['total_retrieved']
                                    all_metrics['full_triples'][threshold]['count'] += 1
                            
                            # Tổng hợp metrics cho matching_pairs
                            if 'matching_pairs' in metrics:
                                all_metrics['matching_pairs']['total_retrieved'] += metrics['matching_pairs']['total_retrieved']
                                all_metrics['matching_pairs']['unique_images'] += metrics['matching_pairs']['unique_images']
                except Exception as e:
                    logger.error(f"Lỗi khi đọc metrics từ {result_folder}: {str(e)}")
    
    # Tính trung bình cho các metrics
    for category in ['pairs', 'full_triples']:
        for threshold, metrics in all_metrics[category].items():
            if metrics['count'] > 0:
                metrics['true_positives'] /= metrics['count']
                metrics['false_positives'] /= metrics['count']
                metrics['false_negatives'] /= metrics['count']
                metrics['true_negatives'] /= metrics['count']
                metrics['total_retrieved'] /= metrics['count']
                
                # Tính lại precision, recall, f1_score
                tp = metrics['true_positives']
                fp = metrics['false_positives']
                fn = metrics['false_negatives']
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1_score'] = f1_score
    
    # Vẽ biểu đồ ROC và Precision-Recall tổng hợp
    plot_aggregate_roc_curve(all_metrics, os.path.join(aggregate_folder, "aggregate_roc_curve.png"))
    plot_aggregate_precision_recall_curve(all_metrics, os.path.join(aggregate_folder, "aggregate_precision_recall_curve.png"))
    
    # Vẽ biểu đồ ROC và Precision-Recall với từng điểm là một ảnh
    plot_individual_roc_curve(individual_metrics, os.path.join(aggregate_folder, "individual_roc_curve.png"))
    plot_individual_precision_recall_curve(individual_metrics, os.path.join(aggregate_folder, "individual_precision_recall_curve.png"))
    
    # Lưu metrics tổng hợp
    with open(os.path.join(aggregate_folder, "aggregate_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # Lưu metrics của từng ảnh
    with open(os.path.join(aggregate_folder, "individual_metrics.json"), "w") as f:
        json.dump(individual_metrics, f, indent=2)
    
    # Lưu log tổng hợp
    with open(os.path.join(aggregate_folder, "aggregate_log.txt"), "w") as f:
        f.write(f"Tổng hợp kết quả từ {len(results)} ảnh\n")
        f.write(f"Thời gian: {timestamp}\n\n")
        
        f.write("Kết quả đánh giá mô hình (trung bình):\n")
        f.write("  Truy vấn theo cặp:\n")
        for threshold, m in all_metrics['pairs'].items():
            f.write(f"    {threshold}:\n")
            f.write(f"      Precision: {m['precision']:.4f}\n")
            f.write(f"      Recall: {m['recall']:.4f}\n")
            f.write(f"      F1-score: {m['f1_score']:.4f}\n")
            f.write(f"      Số ảnh truy xuất trung bình: {m['total_retrieved']:.2f}\n")
        
        f.write("\n  Truy vấn theo bộ ba:\n")
        for threshold, m in all_metrics['full_triples'].items():
            f.write(f"    {threshold}:\n")
            f.write(f"      Precision: {m['precision']:.4f}\n")
            f.write(f"      Recall: {m['recall']:.4f}\n")
            f.write(f"      F1-score: {m['f1_score']:.4f}\n")
            f.write(f"      Số ảnh truy xuất trung bình: {m['total_retrieved']:.2f}\n")
        
        f.write("\n  Matching pairs:\n")
        f.write(f"    Tổng số ảnh truy xuất: {all_metrics['matching_pairs']['total_retrieved']}\n")
        f.write(f"    Số ảnh duy nhất: {all_metrics['matching_pairs']['unique_images']}\n")
    
    logger.info(f"Đã tổng hợp kết quả từ {len(results)} ảnh và lưu vào {aggregate_folder}")
    return results, aggregate_folder

def plot_aggregate_roc_curve(metrics, output_path):
    """
    Vẽ biểu đồ ROC curve tổng hợp từ các số liệu đã tính
    
    Args:
        metrics: Dictionary chứa các số liệu tổng hợp
        output_path: Đường dẫn để lưu biểu đồ
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(10, 8))
        
        # Vẽ ROC curve cho các phương pháp truy vấn theo cặp
        for threshold, m in metrics['pairs'].items():
            tpr = m['recall']  # True Positive Rate = Recall
            fpr = m['false_positives'] / (m['false_positives'] + m['true_negatives']) if (m['false_positives'] + m['true_negatives']) > 0 else 0
            plt.plot(fpr, tpr, 'o-', label=f'Pairs {threshold}')
        
        # Vẽ ROC curve cho các phương pháp truy vấn theo bộ ba
        for threshold, m in metrics['full_triples'].items():
            tpr = m['recall']
            fpr = m['false_positives'] / (m['false_positives'] + m['true_negatives']) if (m['false_positives'] + m['true_negatives']) > 0 else 0
            plt.plot(fpr, tpr, 's-', label=f'Full Triples {threshold}')
        
        # Vẽ đường chéo ngẫu nhiên
        plt.plot([0, 1], [0, 1], 'k--')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Aggregate ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ ROC tổng hợp: {str(e)}")
        return False

def plot_aggregate_precision_recall_curve(metrics, output_path):
    """
    Vẽ biểu đồ Precision-Recall curve tổng hợp từ các số liệu đã tính
    
    Args:
        metrics: Dictionary chứa các số liệu tổng hợp
        output_path: Đường dẫn để lưu biểu đồ
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(10, 8))
        
        # Vẽ Precision-Recall curve cho các phương pháp truy vấn theo cặp
        for threshold, m in metrics['pairs'].items():
            precision = m['precision']
            recall = m['recall']
            plt.plot(recall, precision, 'o-', label=f'Pairs {threshold}')
        
        # Vẽ Precision-Recall curve cho các phương pháp truy vấn theo bộ ba
        for threshold, m in metrics['full_triples'].items():
            precision = m['precision']
            recall = m['recall']
            plt.plot(recall, precision, 's-', label=f'Full Triples {threshold}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Aggregate Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ Precision-Recall tổng hợp: {str(e)}")
        return False

def plot_individual_roc_curve(metrics, output_path):
    """
    Vẽ biểu đồ ROC curve với từng điểm là một ảnh
    
    Args:
        metrics: Dictionary chứa các số liệu của từng ảnh
        output_path: Đường dẫn để lưu biểu đồ
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(12, 10))
        
        # Màu sắc cho các ngưỡng khác nhau
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        markers = ['o', 's', '^', 'D', 'v', 'p', '*']
        
        # Vẽ ROC curve cho các phương pháp truy vấn theo cặp
        for i, (threshold, images) in enumerate(metrics['pairs'].items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Vẽ điểm cho từng ảnh
            for img in images:
                tpr = img['recall']  # True Positive Rate = Recall
                fpr = img['false_positives'] / (img['false_positives'] + img['true_negatives']) if (img['false_positives'] + img['true_negatives']) > 0 else 0
                plt.scatter(fpr, tpr, color=color, marker=marker, s=50, alpha=0.7, label=f'Pairs {threshold}' if img == images[0] else "")
            
            # Vẽ đường trung bình
            avg_tpr = np.mean([img['recall'] for img in images])
            avg_fpr = np.mean([img['false_positives'] / (img['false_positives'] + img['true_negatives']) if (img['false_positives'] + img['true_negatives']) > 0 else 0 for img in images])
            plt.plot(avg_fpr, avg_tpr, color=color, linestyle='--', linewidth=2)
        
        # Vẽ ROC curve cho các phương pháp truy vấn theo bộ ba
        for i, (threshold, images) in enumerate(metrics['full_triples'].items()):
            color = colors[(i + len(metrics['pairs'])) % len(colors)]
            marker = markers[(i + len(metrics['pairs'])) % len(markers)]
            
            # Vẽ điểm cho từng ảnh
            for img in images:
                tpr = img['recall']
                fpr = img['false_positives'] / (img['false_positives'] + img['true_negatives']) if (img['false_positives'] + img['true_negatives']) > 0 else 0
                plt.scatter(fpr, tpr, color=color, marker=marker, s=50, alpha=0.7, label=f'Full Triples {threshold}' if img == images[0] else "")
            
            # Vẽ đường trung bình
            avg_tpr = np.mean([img['recall'] for img in images])
            avg_fpr = np.mean([img['false_positives'] / (img['false_positives'] + img['true_negatives']) if (img['false_positives'] + img['true_negatives']) > 0 else 0 for img in images])
            plt.plot(avg_fpr, avg_tpr, color=color, linestyle='--', linewidth=2)
        
        # Vẽ đường chéo ngẫu nhiên
        plt.plot([0, 1], [0, 1], 'k--')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Individual ROC Curve (Each Point is an Image)')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ ROC với từng điểm là một ảnh: {str(e)}")
        return False

def plot_individual_precision_recall_curve(metrics, output_path):
    """
    Vẽ biểu đồ Precision-Recall curve với từng điểm là một ảnh
    
    Args:
        metrics: Dictionary chứa các số liệu của từng ảnh
        output_path: Đường dẫn để lưu biểu đồ
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(12, 10))
        
        # Màu sắc cho các ngưỡng khác nhau
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        markers = ['o', 's', '^', 'D', 'v', 'p', '*']
        
        # Vẽ Precision-Recall curve cho các phương pháp truy vấn theo cặp
        for i, (threshold, images) in enumerate(metrics['pairs'].items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Vẽ điểm cho từng ảnh
            for img in images:
                precision = img['precision']
                recall = img['recall']
                plt.scatter(recall, precision, color=color, marker=marker, s=50, alpha=0.7, label=f'Pairs {threshold}' if img == images[0] else "")
            
            # Vẽ đường trung bình
            avg_precision = np.mean([img['precision'] for img in images])
            avg_recall = np.mean([img['recall'] for img in images])
            plt.plot(avg_recall, avg_precision, color=color, linestyle='--', linewidth=2)
        
        # Vẽ Precision-Recall curve cho các phương pháp truy vấn theo bộ ba
        for i, (threshold, images) in enumerate(metrics['full_triples'].items()):
            color = colors[(i + len(metrics['pairs'])) % len(colors)]
            marker = markers[(i + len(metrics['pairs'])) % len(markers)]
            
            # Vẽ điểm cho từng ảnh
            for img in images:
                precision = img['precision']
                recall = img['recall']
                plt.scatter(recall, precision, color=color, marker=marker, s=50, alpha=0.7, label=f'Full Triples {threshold}' if img == images[0] else "")
            
            # Vẽ đường trung bình
            avg_precision = np.mean([img['precision'] for img in images])
            avg_recall = np.mean([img['recall'] for img in images])
            plt.plot(avg_recall, avg_precision, color=color, linestyle='--', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Individual Precision-Recall Curve (Each Point is an Image)')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ Precision-Recall với từng điểm là một ảnh: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tự động truy vấn dữ liệu ảnh và lưu kết quả")
    parser.add_argument("--image", help="Đường dẫn đến ảnh cần xử lý")
    parser.add_argument("--directory", help="Đường dẫn đến thư mục chứa ảnh cần xử lý")
    
    args = parser.parse_args()
    
    if args.image:
        result_folder = process_image(args.image)
        if result_folder:
            print(f"Đã xử lý ảnh và lưu kết quả vào {result_folder}")
        else:
            print("Có lỗi xảy ra khi xử lý ảnh")
    elif args.directory:
        result_folders, aggregate_folder = process_directory(args.directory)
        print(f"Đã xử lý {len(result_folders)} ảnh và lưu kết quả vào các thư mục:")
        for folder in result_folders:
            print(f"- {folder}")
        print(f"Kết quả tổng hợp được lưu vào: {aggregate_folder}")
    else:
        print("Vui lòng cung cấp đường dẫn đến ảnh hoặc thư mục cần xử lý") 