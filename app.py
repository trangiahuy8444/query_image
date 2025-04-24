import os
import json
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from neo4j import GraphDatabase
from RelTR.inference import load_model, predict
import time
import shutil
from datetime import datetime

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Cấu hình thư mục
IMAGE_FOLDER = './data/vg_focused/images'  # Thư mục chứa hình ảnh
UPLOAD_FOLDER = './uploads'  # Thư mục để lưu ảnh upload
OUTPUT_FOLDER = './output_images'  # Thư mục để lưu ảnh output
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Đã lưu file {filename} vào {filepath}")

        prediction_start_time = time.time()
        try:
            logger.info("Bắt đầu dự đoán...")
            predictions = predict(filepath, model)
            logger.info(f"Dự đoán thành công: {len(predictions)} kết quả")
            logger.info(f"Chi tiết dự đoán: {json.dumps(predictions, indent=2)}")
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán: {str(e)}")
            return jsonify({"error": f"Lỗi khi dự đoán: {str(e)}"}), 500

        prediction_time = time.time() - prediction_start_time
        logger.info(f"Thời gian dự đoán: {prediction_time:.2f} giây")

        subjects = set(p['subject']['class'] for p in predictions)
        objects = set(p['object']['class'] for p in predictions)
        logger.info(f"Subjects: {subjects}")
        logger.info(f"Objects: {objects}")

        try:
            logger.info("Bắt đầu truy vấn ảnh...")
            matching_pairs_results = find_images_with_matching_pairs(predictions)
            logger.info(f"Truy vấn thành công: {len(matching_pairs_results)} kết quả")
        except Exception as e:
            logger.error(f"Lỗi khi truy vấn ảnh: {str(e)}")
            return jsonify({"error": f"Lỗi khi truy vấn ảnh: {str(e)}"}), 500

        try:
            logger.info("Bắt đầu truy vấn ảnh theo cặp...")
            related_images = {
                "1_or_more": query_images_by_pairs_count(predictions, 1),
                "2_or_more": query_images_by_pairs_count(predictions, 2),
                "3_or_more": query_images_by_pairs_count(predictions, 3),
                "4_or_more": query_images_by_pairs_count(predictions, 4),
                "5_or_more": query_images_by_pairs_count(predictions, 5),
            }
            logger.info(f"Truy vấn theo cặp thành công: {json.dumps({k: len(v) for k, v in related_images.items()}, indent=2)}")
        except Exception as e:
            logger.error(f"Lỗi khi truy vấn ảnh theo cặp: {str(e)}")
            return jsonify({"error": f"Lỗi khi truy vấn ảnh theo cặp: {str(e)}"}), 500

        try:
            logger.info("Bắt đầu truy vấn ảnh theo bộ ba...")
            related_images_full = {
                "1_or_more_full": query_images_by_full_pairs_count(predictions, 1),
                "2_or_more_full": query_images_by_full_pairs_count(predictions, 2),
                "3_or_more_full": query_images_by_full_pairs_count(predictions, 3),
                "4_or_more_full": query_images_by_full_pairs_count(predictions, 4),
                "5_or_more_full": query_images_by_full_pairs_count(predictions, 5),
            }
            logger.info(f"Truy vấn theo bộ ba thành công: {json.dumps({k: len(v) for k, v in related_images_full.items()}, indent=2)}")
        except Exception as e:
            logger.error(f"Lỗi khi truy vấn ảnh theo bộ ba: {str(e)}")
            return jsonify({"error": f"Lỗi khi truy vấn ảnh theo bộ ba: {str(e)}"}), 500

        input_image_name = os.path.splitext(filename)[0]
        output_folder = os.path.join(app.config['OUTPUT_FOLDER'], input_image_name)
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Đã tạo thư mục output: {output_folder}")

        saved_images = set()

        for category, images in related_images.items():
            for image in images:
                if image['image_id'] not in saved_images:
                    image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{image['image_id']}.jpg")
                    if os.path.exists(image_path):
                        shutil.copy2(image_path, os.path.join(output_folder, f"{image['image_id']}.jpg"))
                        saved_images.add(image['image_id'])
                        logger.info(f"Đã lưu ảnh: {image['image_id']}")

        for category, images in related_images_full.items():
            for image in images:
                if image['image_id'] not in saved_images:
                    image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{image['image_id']}.jpg")
                    if os.path.exists(image_path):
                        shutil.copy2(image_path, os.path.join(output_folder, f"{image['image_id']}.jpg"))
                        saved_images.add(image['image_id'])
                        logger.info(f"Đã lưu ảnh: {image['image_id']}")

        # Đếm số lượng ảnh thực tế có thể hiển thị
        counted_images = set()
        total_images = 0
        
        for category in related_images:
            category_total = 0
            for image in related_images[category]:
                image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{image['image_id']}.jpg")
                if os.path.exists(image_path):
                    image["url"] = f"data/vg_focused/images/{image['image_id']}.jpg"
                    category_total += 1
                    
                    if image['image_id'] not in counted_images:
                        counted_images.add(image['image_id'])
                        total_images += 1

        for category in related_images_full:
            category_total = 0
            for image in related_images_full[category]:
                image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{image['image_id']}.jpg")
                if os.path.exists(image_path):
                    image["url"] = f"data/vg_focused/images/{image['image_id']}.jpg"
                    category_total += 1
                    
                    if image['image_id'] not in counted_images:
                        counted_images.add(image['image_id'])
                        total_images += 1

        input_image_id = os.path.splitext(filename)[0]

        # Calculate metrics after getting query results
        total_images_in_db = 108077  # This should be replaced with actual count from database
        metrics = calculate_metrics(predictions, {
            'related_images': related_images,
            'related_images_full': related_images_full,
            'matching_pairs_results': matching_pairs_results
        }, total_images_in_db)
        
        return jsonify({
            "predictions": predictions,
            "matching_pairs_results": matching_pairs_results,
            "related_images": related_images,
            "related_images_full": related_images_full,
            "metrics": {
                "prediction_time": round(prediction_time, 2),
                "num_predictions": len(predictions),
                "total_unique_images": total_images,
                "input_image_id": input_image_id,
                "output_folder": output_folder,
                "roc_pr_metrics": metrics  # Add the new metrics to the response
            }
        })

    except Exception as e:
        logger.error(f"Lỗi không xác định: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/data/vg_focused/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

def get_image_url(image_id):
    return f'data/vg_focused/images/{image_id}.jpg'

def get_related_images_by_subject_object_pairs(graph, subject_classes, object_classes, min_matches=3):
    for image in results:
        image['url'] = get_image_url(image['image_id'])
    return results

def get_related_images_by_full_triples(graph, subject_classes, relation_classes, object_classes, min_matches=3):
    for image in results:
        image['url'] = get_image_url(image['image_id'])
    return results

def get_matching_pairs_results(graph, subject_classes, relation_classes, object_classes):
    for images in matching_pairs_results.values():
        for image in images:
            image['url'] = get_image_url(image['image_id'])
    return matching_pairs_results

def calculate_metrics(predictions, query_results, total_images_in_db):
    """
    Calculate ROC and precision-recall metrics from query results
    
    Args:
        predictions: List of predicted relationships
        query_results: Dictionary containing query results from different methods
        total_images_in_db: Total number of images in the database
    
    Returns:
        Dictionary containing various metrics for ROC and precision-recall analysis
    """
    metrics = {}
    
    # Process results from query_images_by_pairs_count
    for threshold, results in query_results.get('related_images', {}).items():
        tp = len(results)  # True positives - images correctly retrieved
        fp = 0  # False positives - would need ground truth to calculate
        fn = 0  # False negatives - would need ground truth to calculate
        tn = total_images_in_db - tp  # True negatives - remaining images
        
        # Calculate basic metrics
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
    
    # Process results from query_images_by_full_pairs_count
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
    
    # Process results from find_images_with_matching_pairs
    matching_pairs_results = query_results.get('matching_pairs_results', {})
    total_matching_pairs = sum(len(images) for images in matching_pairs_results.values())
    
    metrics['matching_pairs'] = {
        'total_retrieved': total_matching_pairs,
        'unique_images': len(set(img['image_id'] for images in matching_pairs_results.values() for img in images))
    }
    
    return metrics

if __name__ == '__main__':
    app.run(debug=True, port=5002)