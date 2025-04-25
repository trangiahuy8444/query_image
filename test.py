import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from neo4j import GraphDatabase
from RelTR.inference import load_model, predict
import time
import json

# Kết nối Neo4j
# uri = "bolt://localhost:7689"
uri = "neo4j+s://b40b4f2a.databases.neo4j.io"
username = "neo4j"
password = "fpKNUXKT-4z0kQMm1nuUaiXe8p70uIebc3y3a4Z8kUA"
# password = "12345678"
driver = GraphDatabase.driver(uri, auth=(username, password))

def load_model_and_predict(image_path, model_path):
    """
    Tải mô hình và thực hiện dự đoán trên ảnh
    
    Args:
        image_path: Đường dẫn đến ảnh cần dự đoán
        model_path: Đường dẫn đến file checkpoint của mô hình
        
    Returns:
        predictions: Kết quả dự đoán từ mô hình
    """
    model = load_model(model_path)
    predictions = predict(image_path, model)
    return predictions

def get_predictions_from_model(predictions):
    """
    Chuyển đổi kết quả dự đoán từ mô hình thành định dạng chuẩn
    
    Args:
        predictions: Kết quả dự đoán từ mô hình
        
    Returns:
        predicted_triplets: Danh sách các triplet dự đoán (subject, relation, object)
    """
    predicted_triplets = []
    for p in predictions:
        predicted_triplets.append({
            'subject': p['subject']['class'],
            'relation': p['relation']['class'],
            'object': p['object']['class']
        })
    return predicted_triplets

def query_images_by_pairs(predictions, min_pairs, image_folder='./data/vg_focused/images'):
    """
    Truy vấn ảnh từ Neo4j dựa trên các cặp subject-object
    
    Args:
        predictions: Kết quả dự đoán từ mô hình
        min_pairs: Số lượng cặp tối thiểu cần khớp
        image_folder: Thư mục chứa ảnh
        
    Returns:
        image_details: Danh sách các ảnh và thông tin chi tiết
    """
    image_details = []
    processed_image_ids = set()
    query_start_time = time.time()
    
    with driver.session() as session:
        # Tạo set các subject và object từ dự đoán
        subjects = set(p['subject']['class'] for p in predictions)
        objects = set(p['object']['class'] for p in predictions)
        prediction_pairs = [(p['subject']['class'], p['object']['class']) for p in predictions]
        
        # Truy vấn Neo4j để lấy các ảnh và cặp subject-object
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
        
        # Gửi tham số cho truy vấn
        params = {
            'subjects': list(subjects),
            'objects': list(objects),
            'prediction_pairs': prediction_pairs,
            'min_pairs': min_pairs
        }
        
        result = session.run(query, params)
        
        for record in result:
            image_id = record['image_id']
            
            # Kiểm tra sự tồn tại của file ảnh
            image_path = os.path.join(image_folder, f"{image_id}.jpg")
            if not os.path.exists(image_path) or image_id in processed_image_ids:
                continue
                
            processed_image_ids.add(image_id)
            relationships_str = []
            for rel in record['relationships']:
                relationships_str.append(
                    f"{rel['subject']} - {rel['relation']} - {rel['object']}"
                )
            
            image_details.append({
                "image_id": image_id,
                "matching_pairs": record['matching_pairs'],
                "total_pairs": record['total_pairs'],
                "matching_percentage": round(record['matching_percentage'], 2),
                "relationships": relationships_str
            })
    
    query_time = time.time() - query_start_time
    print(f"Query time: {query_time:.2f} seconds")
    
    return image_details

def query_images_triplets(predictions, min_pairs, image_folder='./data/vg_focused/images'):
    """
    Truy vấn ảnh từ Neo4j dựa trên các triplet subject-relation-object
    
    Args:
        predictions: Kết quả dự đoán từ mô hình
        min_pairs: Số lượng triplet tối thiểu cần khớp
        image_folder: Thư mục chứa ảnh
        
    Returns:
        image_details: Danh sách các ảnh và thông tin chi tiết
    """
    image_details = []
    processed_image_ids = set()
    query_start_time = time.time()
    
    with driver.session() as session:
        # Tạo set các subject, relation và object từ dự đoán
        subjects = set(p['subject']['class'] for p in predictions)
        relations = set(p['relation']['class'] for p in predictions)
        objects = set(p['object']['class'] for p in predictions)
        prediction_triples = [(p['subject']['class'], p['relation']['class'], p['object']['class']) for p in predictions]
        
        # Truy vấn Neo4j để lấy các ảnh và bộ ba subject-relation-object
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
        
        # Gửi tham số cho truy vấn
        params = {
            'subjects': list(subjects),
            'objects': list(objects),
            'prediction_triples': prediction_triples,
            'min_pairs': min_pairs
        }
        
        result = session.run(query, params)
        
        for record in result:
            image_id = record['image_id']
            
            if image_id in processed_image_ids:
                continue
                
            processed_image_ids.add(image_id)
            relationships_str = []
            for rel in record['relationships']:
                relationships_str.append(
                    f"{rel['subject']} - {rel['relation']} - {rel['object']}"
                )
            
            image_details.append({
                "image_id": image_id,
                "matching_triples": record['matching_triples'],
                "total_triples": record['total_triples'],
                "matching_percentage": round(record['matching_percentage'], 2),
                "relationships": relationships_str
            })
    
    query_time = time.time() - query_start_time
    print(f"Query time: {query_time:.2f} seconds")
    
    return image_details

def calculate_roc_pr_metrics(model_predictions, ground_truth_data):
    """
    Tính toán metrics cho ROC và Precision-Recall curves
    
    Args:
        model_predictions: Danh sách các dự đoán từ mô hình
        ground_truth_data: Danh sách các ground truth từ cơ sở dữ liệu
        
    Returns:
        y_true: Mảng nhãn thực (1: true positive, 0: false positive)
        y_score: Mảng điểm số tương ứng
    """
    # Kiểm tra định dạng của model_predictions
    formatted_predictions = []
    
    # Nếu model_predictions là danh sách các chuỗi
    if isinstance(model_predictions[0], str):
        for pred in model_predictions:
            # Phân tích chuỗi dự đoán "subject - relation - object"
            parts = pred.split(" - ")
            if len(parts) == 3:
                subject, relation, obj = parts
                formatted_predictions.append({
                    'subject': subject,
                    'relation': relation,
                    'object': obj
                })
    # Nếu model_predictions là danh sách các dictionary
    elif isinstance(model_predictions[0], dict):
        formatted_predictions = model_predictions
    
    # Tạo danh sách tất cả các ground truth triplets từ dữ liệu
    all_ground_truth_triplets = []
    for item in ground_truth_data:
        for rel in item['relationships']:
            # Phân tích chuỗi relationship "subject - relation - object"
            parts = rel.split(" - ")
            if len(parts) == 3:
                subject, relation, obj = parts
                all_ground_truth_triplets.append({
                    'subject': subject,
                    'relation': relation,
                    'object': obj
                })
    
    # Tính toán true positives và false positives
    y_true = []
    y_score = []
    
    # Đối với mỗi dự đoán, kiểm tra xem nó có khớp với ground truth không
    for pred in formatted_predictions:
        is_true_positive = False
        for gt in all_ground_truth_triplets:
            if (pred['subject'] == gt['subject'] and 
                pred['relation'] == gt['relation'] and 
                pred['object'] == gt['object']):
                is_true_positive = True
                break
        
        # Thêm vào danh sách kết quả
        y_true.append(1 if is_true_positive else 0)
        # Giả sử điểm số xác suất là 1.0 cho tất cả các dự đoán
        y_score.append(1.0)
    
    # Thêm các ground truth không được dự đoán (false negatives)
    for gt in all_ground_truth_triplets:
        is_predicted = False
        for pred in formatted_predictions:
            if (pred['subject'] == gt['subject'] and 
                pred['relation'] == gt['relation'] and 
                pred['object'] == gt['object']):
                is_predicted = True
                break
        
        if not is_predicted:
            y_true.append(0)  # False negative
            y_score.append(0.0)  # Điểm số thấp cho false negative
    
    return np.array(y_true), np.array(y_score)

def plot_roc_and_pr_curve(y_true, y_score, save_path=None, label=None):
    """
    Vẽ ROC và Precision-Recall curves
    
    Args:
        y_true: Mảng nhãn thực
        y_score: Mảng điểm số
        save_path: Đường dẫn để lưu đồ thị (nếu None thì hiển thị đồ thị)
        label: Nhãn cho đường cong (nếu None thì không hiển thị nhãn)
    """
    # Tính toán ROC curve và AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Tính toán Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    # Vẽ ROC curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if label:
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
    else:
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Vẽ Precision-Recall curve
    plt.subplot(1, 2, 2)
    if label:
        plt.plot(recall, precision, lw=2, label=f'{label} (AP = {average_precision:.2f})')
    else:
        plt.plot(recall, precision, lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_all_curves(image_path, model_path, save_path=None):
    """
    Vẽ 10 đường cong tương ứng với images_pairs 1-5 và images_triplets 1-5
    
    Args:
        image_path: Đường dẫn đến ảnh cần đánh giá
        model_path: Đường dẫn đến file checkpoint của mô hình
        save_path: Đường dẫn để lưu đồ thị (nếu None thì hiển thị đồ thị)
    """
    # Tải mô hình và thực hiện dự đoán
    predictions = load_model_and_predict(image_path, model_path)
    model_predictions = get_predictions_from_model(predictions)
    
    # Khởi tạo danh sách trống cho ground truth
    ground_truth_pairs_by_min = {i: [] for i in range(1, 6)}
    ground_truth_triplets_by_min = {i: [] for i in range(1, 6)}

    # Dùng vòng lặp để truy vấn các pairs và triplets
    for min_pairs in range(1, 6):
        pairs = query_images_by_pairs(predictions, min_pairs)
        triplets = query_images_triplets(predictions, min_pairs)
        ground_truth_pairs_by_min[min_pairs] = pairs
        ground_truth_triplets_by_min[min_pairs] = triplets

    # Vẽ ROC và Precision-Recall curves
    plt.figure(figsize=(15, 10))
    
    # Vẽ ROC curves
    plt.subplot(2, 2, 1)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Vẽ Precision-Recall curves
    plt.subplot(2, 2, 2)
    
    # Màu sắc cho các đường cong
    colors = ['red', 'blue', 'green', 'purple', 'orange', 
              'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Vẽ các đường cong cho pairs
    for min_pairs in range(1, 6):
        # Tính toán metrics
        y_true, y_score = calculate_roc_pr_metrics(model_predictions, ground_truth_pairs_by_min[min_pairs])
        
        # Tính ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Tính Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
        
        # Vẽ ROC curve
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, color=colors[min_pairs-1], lw=2, 
                 label=f'Pairs (min={min_pairs}, AUC={roc_auc:.2f})')
        
        # Vẽ Precision-Recall curve
        plt.subplot(2, 2, 2)
        plt.plot(recall, precision, color=colors[min_pairs-1], lw=2, 
                 label=f'Pairs (min={min_pairs}, AP={average_precision:.2f})')
    
    # Vẽ các đường cong cho triplets
    for min_triplets in range(1, 6):
        # Tính toán metrics
        y_true, y_score = calculate_roc_pr_metrics(model_predictions, ground_truth_triplets_by_min[min_triplets])
        
        # Tính ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Tính Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
        
        # Vẽ ROC curve
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, color=colors[min_triplets+4], lw=2, linestyle='--', 
                 label=f'Triplets (min={min_triplets}, AUC={roc_auc:.2f})')
        
        # Vẽ Precision-Recall curve
        plt.subplot(2, 2, 2)
        plt.plot(recall, precision, color=colors[min_triplets+4], lw=2, linestyle='--', 
                 label=f'Triplets (min={min_triplets}, AP={average_precision:.2f})')
    
    # Hoàn thiện biểu đồ ROC
    plt.subplot(2, 2, 1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize=8)
    
    # Hoàn thiện biểu đồ Precision-Recall
    plt.subplot(2, 2, 2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower left", fontsize=8)
    
    # Vẽ biểu đồ phân phối số lượng ảnh
    plt.subplot(2, 2, 3)
    pairs_counts = [len(ground_truth_pairs_by_min[i]) for i in range(1, 6)]
    triplets_counts = [len(ground_truth_triplets_by_min[i]) for i in range(1, 6)]
    
    x = np.arange(1, 6)
    width = 0.35
    
    plt.bar(x - width/2, pairs_counts, width, label='Pairs', color='blue')
    plt.bar(x + width/2, triplets_counts, width, label='Triplets', color='red')
    
    plt.xlabel('Min Pairs/Triplets')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images by Min Pairs/Triplets')
    plt.xticks(x)
    plt.legend()
    
    # Vẽ biểu đồ phân phối matching percentage
    plt.subplot(2, 2, 4)
    pairs_percentages = [np.mean([img['matching_percentage'] for img in ground_truth_pairs_by_min[i]]) for i in range(1, 6)]
    triplets_percentages = [np.mean([img['matching_percentage'] for img in ground_truth_triplets_by_min[i]]) for i in range(1, 6)]
    
    plt.bar(x - width/2, pairs_percentages, width, label='Pairs', color='blue')
    plt.bar(x + width/2, triplets_percentages, width, label='Triplets', color='red')
    
    plt.xlabel('Min Pairs/Triplets')
    plt.ylabel('Average Matching Percentage')
    plt.title('Average Matching Percentage by Min Pairs/Triplets')
    plt.xticks(x)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return {
        'pairs_counts': pairs_counts,
        'triplets_counts': triplets_counts,
        'pairs_percentages': pairs_percentages,
        'triplets_percentages': triplets_percentages
    }

def evaluate_model(image_path, model_path, min_pairs_range=(1, 6), save_results=True):
    """
    Đánh giá mô hình trên một ảnh cụ thể
    
    Args:
        image_path: Đường dẫn đến ảnh cần đánh giá
        model_path: Đường dẫn đến file checkpoint của mô hình
        min_pairs_range: Tuple (start, end) cho range của min_pairs
        save_results: Nếu True, lưu kết quả vào file JSON
        
    Returns:
        results: Dictionary chứa kết quả đánh giá
    """
    # Tải mô hình và thực hiện dự đoán
    predictions = load_model_and_predict(image_path, model_path)
    model_predictions = get_predictions_from_model(predictions)
    
    # Khởi tạo danh sách trống cho ground truth
    ground_truth_pairs = []
    ground_truth_triplets = []

    # Dùng vòng lặp để truy vấn các pairs và triplets
    for min_pairs in range(min_pairs_range[0], min_pairs_range[1]):
        pairs = query_images_by_pairs(predictions, min_pairs)
        triplets = query_images_triplets(predictions, min_pairs)
        ground_truth_pairs.extend(pairs)
        ground_truth_triplets.extend(triplets)

    # Lưu kết quả vào file JSON nếu cần
    if save_results:
        with open("ground_truth_pairs.json", "w") as f:
            json.dump(ground_truth_pairs, f, indent=4)
        with open("ground_truth_triplets.json", "w") as f:
            json.dump(ground_truth_triplets, f, indent=4)
    
    # Đánh giá mô hình với dữ liệu pairs
    pairs_metrics = evaluate_model_with_data(model_predictions, ground_truth_pairs, save_plots=False)
    
    # Đánh giá mô hình với dữ liệu triplets
    triplets_metrics = evaluate_model_with_data(model_predictions, ground_truth_triplets, save_plots=False)
    
    return {
        'pairs_metrics': pairs_metrics,
        'triplets_metrics': triplets_metrics,
        'ground_truth_pairs': ground_truth_pairs,
        'ground_truth_triplets': ground_truth_triplets
    }

def evaluate_model_with_data(model_predictions, ground_truth_data, threshold=0.5, save_plots=False):
    """
    Đánh giá mô hình với dữ liệu dự đoán và ground truth cụ thể
    
    Args:
        model_predictions: Danh sách các dự đoán từ mô hình
        ground_truth_data: Danh sách các ground truth từ cơ sở dữ liệu
        threshold: Ngưỡng phân loại (mặc định: 0.5)
        save_plots: Nếu True, lưu đồ thị thay vì hiển thị
        
    Returns:
        metrics: Dictionary chứa các metrics đánh giá
    """
    # Tính toán metrics
    y_true, y_score = calculate_roc_pr_metrics(model_predictions, ground_truth_data)
    
    # Vẽ ROC và Precision-Recall curves chỉ khi save_plots=True
    if save_plots:
        plot_roc_and_pr_curve(y_true, y_score, save_path="roc_pr_curves.png")
    
    # Chuyển đổi y_true thành nhãn nhị phân (0 hoặc 1)
    y_true_binary = (y_true > 0).astype(int)
    
    # Tính precision, recall, và F1 score
    precision = precision_score(y_true_binary, y_score > threshold)
    recall = recall_score(y_true_binary, y_score > threshold)
    f1 = f1_score(y_true_binary, y_score > threshold)
    
    return {
        'precision': float(precision),  # Chuyển đổi thành float để JSON serializable
        'recall': float(recall),        # Chuyển đổi thành float để JSON serializable
        'f1': float(f1),                # Chuyển đổi thành float để JSON serializable
        'y_true': y_true.tolist(),      # Chuyển đổi NumPy array thành list
        'y_score': y_score.tolist()     # Chuyển đổi NumPy array thành list
    }

def evaluate_model_on_dataset(image_folder, model_path, min_pairs_range=(1, 6), save_results=True, max_images=None):
    """
    Đánh giá mô hình trên toàn bộ bộ dữ liệu ảnh
    
    Args:
        image_folder: Thư mục chứa ảnh cần đánh giá
        model_path: Đường dẫn đến file checkpoint của mô hình
        min_pairs_range: Tuple (start, end) cho range của min_pairs
        save_results: Nếu True, lưu kết quả vào file JSON
        max_images: Số lượng ảnh tối đa cần đánh giá (None nếu đánh giá tất cả)
        
    Returns:
        results: Dictionary chứa kết quả đánh giá
    """
    # Lấy danh sách tất cả các ảnh trong thư mục
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Giới hạn số lượng ảnh nếu cần
    if max_images is not None and max_images < len(image_files):
        image_files = image_files[:max_images]
    
    print(f"Đánh giá mô hình trên {len(image_files)} ảnh...")
    
    # Khởi tạo danh sách trống cho kết quả
    all_results = []
    
    # Đánh giá từng ảnh
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        print(f"Đánh giá ảnh {i+1}/{len(image_files)}: {image_file}")
        
        try:
            # Đánh giá mô hình trên ảnh hiện tại
            result = evaluate_model(image_path, model_path, min_pairs_range, save_results=False)
            
            # Thêm thông tin ảnh vào kết quả
            result['image_file'] = image_file
            all_results.append(result)
            
        except Exception as e:
            print(f"Lỗi khi đánh giá ảnh {image_file}: {str(e)}")
    
    # Tính toán metrics trung bình
    avg_metrics = {
        'pairs': {
            'precision': float(np.mean([r['pairs_metrics']['precision'] for r in all_results])),
            'recall': float(np.mean([r['pairs_metrics']['recall'] for r in all_results])),
            'f1': float(np.mean([r['pairs_metrics']['f1'] for r in all_results]))
        },
        'triplets': {
            'precision': float(np.mean([r['triplets_metrics']['precision'] for r in all_results])),
            'recall': float(np.mean([r['triplets_metrics']['recall'] for r in all_results])),
            'f1': float(np.mean([r['triplets_metrics']['f1'] for r in all_results]))
        }
    }
    
    # In kết quả trung bình
    print("\nKết quả trung bình trên toàn bộ bộ dữ liệu:")
    print(f"Pairs - Precision: {avg_metrics['pairs']['precision']:.4f}, Recall: {avg_metrics['pairs']['recall']:.4f}, F1: {avg_metrics['pairs']['f1']:.4f}")
    print(f"Triplets - Precision: {avg_metrics['triplets']['precision']:.4f}, Recall: {avg_metrics['triplets']['recall']:.4f}, F1: {avg_metrics['triplets']['f1']:.4f}")
    
    # Chuẩn bị dữ liệu để lưu vào JSON
    json_safe_results = {
        'average_metrics': avg_metrics,
        'individual_results': []
    }
    
    # Chuyển đổi kết quả từng ảnh thành định dạng JSON-safe
    for result in all_results:
        json_safe_result = {
            'image_file': result['image_file'],
            'pairs_metrics': {
                'precision': float(result['pairs_metrics']['precision']),
                'recall': float(result['pairs_metrics']['recall']),
                'f1': float(result['pairs_metrics']['f1'])
            },
            'triplets_metrics': {
                'precision': float(result['triplets_metrics']['precision']),
                'recall': float(result['triplets_metrics']['recall']),
                'f1': float(result['triplets_metrics']['f1'])
            }
        }
        json_safe_results['individual_results'].append(json_safe_result)
    
    # Lưu kết quả vào file JSON nếu cần
    if save_results:
        with open("dataset_evaluation_results.json", "w") as f:
            json.dump(json_safe_results, f, indent=4)
    
    return {
        'average_metrics': avg_metrics,
        'individual_results': all_results
    }

def plot_dataset_curves(dataset_results, save_path=None):
    """
    Vẽ biểu đồ ROC và Precision-Recall từ kết quả đánh giá trên bộ dữ liệu
    
    Args:
        dataset_results: Kết quả đánh giá từ hàm evaluate_model_on_dataset
        save_path: Đường dẫn để lưu đồ thị (nếu None thì hiển thị đồ thị)
    """
    # Trích xuất dữ liệu
    individual_results = dataset_results['individual_results']
    
    # Tạo biểu đồ
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ ROC
    plt.subplot(1, 2, 1)
    
    # Vẽ đường ROC cho pairs
    pairs_y_true = []
    pairs_y_score = []
    for result in individual_results:
        pairs_y_true.extend(result['pairs_metrics']['y_true'])
        pairs_y_score.extend(result['pairs_metrics']['y_score'])
    
    fpr, tpr, _ = roc_curve(pairs_y_true, pairs_y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Pairs (AUC = {roc_auc:.2f})')
    
    # Vẽ đường ROC cho triplets
    triplets_y_true = []
    triplets_y_score = []
    for result in individual_results:
        triplets_y_true.extend(result['triplets_metrics']['y_true'])
        triplets_y_score.extend(result['triplets_metrics']['y_score'])
    
    fpr, tpr, _ = roc_curve(triplets_y_true, triplets_y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='red', lw=2, label=f'Triplets (AUC = {roc_auc:.2f})')
    
    # Vẽ đường chéo ngẫu nhiên
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Biểu đồ Precision-Recall
    plt.subplot(1, 2, 2)
    
    # Vẽ đường Precision-Recall cho pairs
    precision, recall, _ = precision_recall_curve(pairs_y_true, pairs_y_score)
    average_precision = average_precision_score(pairs_y_true, pairs_y_score)
    plt.plot(recall, precision, color='blue', lw=2, label=f'Pairs (AP = {average_precision:.2f})')
    
    # Vẽ đường Precision-Recall cho triplets
    precision, recall, _ = precision_recall_curve(triplets_y_true, triplets_y_score)
    average_precision = average_precision_score(triplets_y_true, triplets_y_score)
    plt.plot(recall, precision, color='red', lw=2, label=f'Triplets (AP = {average_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Thực thi đánh giá mô hình
if __name__ == "__main__":
    # Đường dẫn mặc định
    model_path = './RelTR/ckpt/fine_tune1/checkpoint0049.pth'
    image_folder = "./image_test"  # Thư mục chứa nhiều ảnh
    
    # Đánh giá mô hình trên toàn bộ bộ dữ liệu
    print("Bắt đầu đánh giá mô hình trên nhiều ảnh...")
    dataset_results = evaluate_model_on_dataset(image_folder, model_path, max_images=None)  # Đánh giá tất cả ảnh
    
    # Vẽ biểu đồ tổng hợp từ kết quả đánh giá
    print("Vẽ biểu đồ tổng hợp...")
    plot_dataset_curves(dataset_results, save_path="dataset_curves.png")
    
    # In kết quả chi tiết
    print("\nKết quả đánh giá chi tiết:")
    print(f"Số lượng ảnh đã đánh giá: {len(dataset_results['individual_results'])}")
    print("\nMetrics trung bình:")
    print(f"Pairs - Precision: {dataset_results['average_metrics']['pairs']['precision']:.4f}, "
          f"Recall: {dataset_results['average_metrics']['pairs']['recall']:.4f}, "
          f"F1: {dataset_results['average_metrics']['pairs']['f1']:.4f}")
    print(f"Triplets - Precision: {dataset_results['average_metrics']['triplets']['precision']:.4f}, "
          f"Recall: {dataset_results['average_metrics']['triplets']['recall']:.4f}, "
          f"F1: {dataset_results['average_metrics']['triplets']['f1']:.4f}")
    
    print("\nKết quả đã được lưu vào file 'dataset_evaluation_results.json'")
    print("Biểu đồ tổng hợp đã được lưu vào file 'dataset_curves.png'")