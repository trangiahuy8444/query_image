import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from neo4j import GraphDatabase
from RelTR.inference import load_model, predict
import time
import json
from neo4j.exceptions import ServiceUnavailable
import atexit
import concurrent.futures
import threading

# Kết nối Neo4j với cấu hình tối ưu
# uri = "bolt://localhost:7689"
uri = "neo4j+s://b40b4f2a.databases.neo4j.io"
username = "neo4j"
password = "fpKNUXKT-4z0kQMm1nuUaiXe8p70uIebc3y3a4Z8kUA"
# password = "12345678"

# Cấu hình kết nối tối ưu
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# Tạo driver với cấu hình tối ưu
driver = GraphDatabase.driver(
    uri, 
    auth=(username, password),
    max_connection_lifetime=3600,  # Kết nối tối đa 1 giờ
    max_connection_pool_size=50,   # Số lượng kết nối tối đa trong pool
    connection_acquisition_timeout=60,  # Thời gian chờ kết nối tối đa (giây)
    connection_timeout=30  # Thời gian chờ kết nối ban đầu (giây)
)

# Hàm kiểm tra kết nối
def check_connection():
    try:
        with driver.session() as session:
            result = session.run("RETURN 1")
            return result.single()[0] == 1
    except ServiceUnavailable:
        print("Không thể kết nối đến Neo4j. Vui lòng kiểm tra lại kết nối mạng hoặc thông tin đăng nhập.")
        return False

# Kiểm tra kết nối khi khởi động
if not check_connection():
    print("Cảnh báo: Không thể kết nối đến Neo4j. Các truy vấn có thể sẽ thất bại.")

# Hàm đóng kết nối khi kết thúc chương trình
def close_connection():
    driver.close()
    print("Đã đóng kết nối Neo4j.")

# Đăng ký hàm đóng kết nối khi chương trình kết thúc
atexit.register(close_connection)

def load_model_and_predict(image_path, model_path):
    """
    Tải mô hình và thực hiện dự đoán trên ảnh
    
    Args:
        image_path: Đường dẫn đến ảnh cần dự đoán
        model_path: Đường dẫn đến file checkpoint của mô hình
        
    Returns:
        predictions: Kết quả dự đoán từ mô hình
    """
    try:
        model = load_model(model_path)
        raw_predictions = predict(image_path, model)
        
        # Kiểm tra và chuyển đổi dự đoán thành định dạng chuẩn
        valid_predictions = []
        
        # Nếu không có dự đoán hoặc dự đoán không phải list
        if not raw_predictions or not isinstance(raw_predictions, list):
            print(f"Không có dự đoán hợp lệ cho ảnh {image_path}")
            return []
            
        # Lọc và chuyển đổi dự đoán
        for pred in raw_predictions:
            try:
                # Nếu pred là dictionary, kiểm tra cấu trúc
                if isinstance(pred, dict):
                    if all(k in pred for k in ['subject', 'relation', 'object']):
                        # Đảm bảo các giá trị là dictionary với key 'class'
                        valid_pred = {
                            'subject': {'class': str(pred['subject']) if isinstance(pred['subject'], (str, int)) else pred['subject'].get('class', '')},
                            'relation': {'class': str(pred['relation']) if isinstance(pred['relation'], (str, int)) else pred['relation'].get('class', '')},
                            'object': {'class': str(pred['object']) if isinstance(pred['object'], (str, int)) else pred['object'].get('class', '')}
                        }
                        valid_predictions.append(valid_pred)
                # Nếu pred là tuple/list có 3 phần tử, chuyển thành dictionary
                elif isinstance(pred, (tuple, list)) and len(pred) >= 3:
                    valid_predictions.append({
                        'subject': {'class': str(pred[0])},
                        'relation': {'class': str(pred[1])},
                        'object': {'class': str(pred[2])}
                    })
            except Exception as e:
                print(f"Lỗi khi xử lý dự đoán: {str(e)}, Dữ liệu: {pred}")
                continue
                
        if not valid_predictions:
            print(f"Không tìm thấy dự đoán hợp lệ nào cho ảnh {image_path}")
            
        return valid_predictions
        
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
        return []

def get_predictions_from_model(predictions):
    """
    Chuyển đổi kết quả dự đoán từ mô hình thành định dạng chuẩn
    
    Args:
        predictions: Kết quả dự đoán từ mô hình
        
    Returns:
        predicted_triplets: Danh sách các triplet dự đoán (subject, relation, object)
    """
    predicted_triplets = []
    
    # Kiểm tra nếu predictions không phải là list hoặc rỗng
    if not isinstance(predictions, list):
        print(f"Cảnh báo: Kết quả dự đoán không phải là list: {type(predictions)}")
        return predicted_triplets
        
    if len(predictions) == 0:
        print("Cảnh báo: Không có dự đoán nào")
        return predicted_triplets
    
    for p in predictions:
        try:
            # Kiểm tra cấu trúc của p
            if not isinstance(p, dict):
                print(f"Cảnh báo: Phần tử dự đoán không phải là dictionary: {p}")
                continue
                
            # Kiểm tra và lấy các trường cần thiết
            subject = p.get('subject', {})
            relation = p.get('relation', {})
            obj = p.get('object', {})
            
            # Kiểm tra nếu các trường là dictionary
            if not isinstance(subject, dict) or not isinstance(relation, dict) or not isinstance(obj, dict):
                print(f"Cảnh báo: Cấu trúc dự đoán không đúng định dạng: {p}")
                continue
                
            # Lấy class từ các trường
            subject_class = subject.get('class')
            relation_class = relation.get('class')
            object_class = obj.get('class')
            
            # Kiểm tra nếu có đủ thông tin
            if subject_class and relation_class and object_class:
                predicted_triplets.append({
                    'subject': subject_class,
                    'relation': relation_class,
                    'object': object_class
                })
            else:
                print(f"Cảnh báo: Thiếu thông tin trong dự đoán: {p}")
                
        except Exception as e:
            print(f"Lỗi khi xử lý dự đoán: {e}, Dữ liệu: {p}")
            continue
            
    return predicted_triplets

# Biến để theo dõi số lượng truy vấn đang chạy
active_queries = 0
active_queries_lock = threading.Lock()

def query_images_by_pairs_parallel(predictions_list, min_pairs, max_workers=5, image_folder='./data/vg_focused/images'):
    """
    Truy vấn song song nhiều ảnh từ Neo4j dựa trên các cặp subject-object
    
    Args:
        predictions_list: Danh sách các dự đoán từ mô hình cho nhiều ảnh
        min_pairs: Số lượng cặp tối thiểu cần khớp
        max_workers: Số lượng worker tối đa cho việc xử lý song song
        image_folder: Thư mục chứa ảnh
        
    Returns:
        all_image_details: Danh sách các ảnh và thông tin chi tiết cho tất cả các ảnh
    """
    all_image_details = []
    query_start_time = time.time()
    
    # Hàm worker để xử lý một ảnh
    def process_image(predictions):
        global active_queries
        
        # Tăng số lượng truy vấn đang chạy
        with active_queries_lock:
            active_queries += 1
            current_active = active_queries
            print(f"Đang chạy {current_active} truy vấn song song")
        
        try:
            # Tạo set các subject và object từ dự đoán
            subjects = set(p['subject']['class'] for p in predictions)
            objects = set(p['object']['class'] for p in predictions)
            prediction_pairs = [(p['subject']['class'], p['object']['class']) for p in predictions]
            
            # Truy vấn Neo4j để lấy các ảnh và cặp subject-object - đã tối ưu hóa
            query = """
            // Sử dụng UNWIND để tối ưu hóa việc lọc
            UNWIND $subjects AS subject
            MATCH (s:Object {category: subject})
            WITH DISTINCT s.image_id AS image_id
            
            // Kiểm tra xem có object phù hợp trong cùng ảnh không
            MATCH (o:Object)
            WHERE o.image_id = image_id AND o.category IN $objects
            WITH DISTINCT image_id
            
            // Lấy tất cả các mối quan hệ trong ảnh này
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
            
            image_details = []
            processed_image_ids = set()
            
            with driver.session() as session:
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
            
            return image_details
        finally:
            # Giảm số lượng truy vấn đang chạy
            with active_queries_lock:
                active_queries -= 1
    
    # Sử dụng ThreadPoolExecutor để xử lý song song
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Gửi tất cả các ảnh để xử lý
        future_to_predictions = {executor.submit(process_image, predictions): i for i, predictions in enumerate(predictions_list)}
        
        # Thu thập kết quả khi hoàn thành
        for future in concurrent.futures.as_completed(future_to_predictions):
            image_index = future_to_predictions[future]
            try:
                image_details = future.result()
                all_image_details.extend(image_details)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {image_index}: {str(e)}")
    
    query_time = time.time() - query_start_time
    print(f"Tổng thời gian truy vấn song song: {query_time:.2f} giây")
    
    return all_image_details

def query_images_triplets_parallel(predictions_list, min_pairs, max_workers=5, image_folder='./data/vg_focused/images'):
    """
    Truy vấn song song nhiều ảnh từ Neo4j dựa trên các triplet subject-relation-object
    
    Args:
        predictions_list: Danh sách các dự đoán từ mô hình cho nhiều ảnh
        min_pairs: Số lượng triplet tối thiểu cần khớp
        max_workers: Số lượng worker tối đa cho việc xử lý song song
        image_folder: Thư mục chứa ảnh
        
    Returns:
        all_image_details: Danh sách các ảnh và thông tin chi tiết cho tất cả các ảnh
    """
    all_image_details = []
    query_start_time = time.time()
    
    # Hàm worker để xử lý một ảnh
    def process_image(predictions):
        global active_queries
        
        # Tăng số lượng truy vấn đang chạy
        with active_queries_lock:
            active_queries += 1
            current_active = active_queries
            print(f"Đang chạy {current_active} truy vấn song song")
        
        try:
            # Tạo set các subject, relation và object từ dự đoán
            subjects = set(p['subject']['class'] for p in predictions)
            relations = set(p['relation']['class'] for p in predictions)
            objects = set(p['object']['class'] for p in predictions)
            prediction_triples = [(p['subject']['class'], p['relation']['class'], p['object']['class']) for p in predictions]
            
            # Truy vấn Neo4j để lấy các ảnh và bộ ba subject-relation-object - đã tối ưu hóa
            query = """
            // Sử dụng UNWIND để tối ưu hóa việc lọc
            UNWIND $subjects AS subject
            MATCH (s:Object {category: subject})
            WITH DISTINCT s.image_id AS image_id
            
            // Kiểm tra xem có object phù hợp trong cùng ảnh không
            MATCH (o:Object)
            WHERE o.image_id = image_id AND o.category IN $objects
            WITH DISTINCT image_id
            
            // Lấy tất cả các mối quan hệ trong ảnh này
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
            
            image_details = []
            processed_image_ids = set()
            
            with driver.session() as session:
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
            
            return image_details
        finally:
            # Giảm số lượng truy vấn đang chạy
            with active_queries_lock:
                active_queries -= 1
    
    # Sử dụng ThreadPoolExecutor để xử lý song song
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Gửi tất cả các ảnh để xử lý
        future_to_predictions = {executor.submit(process_image, predictions): i for i, predictions in enumerate(predictions_list)}
        
        # Thu thập kết quả khi hoàn thành
        for future in concurrent.futures.as_completed(future_to_predictions):
            image_index = future_to_predictions[future]
            try:
                image_details = future.result()
                all_image_details.extend(image_details)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {image_index}: {str(e)}")
    
    query_time = time.time() - query_start_time
    print(f"Tổng thời gian truy vấn song song: {query_time:.2f} giây")
    
    return all_image_details

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
        pairs = query_images_by_pairs_parallel([predictions], min_pairs)
        triplets = query_images_triplets_parallel([predictions], min_pairs)
        ground_truth_pairs_by_min[min_pairs] = pairs
        ground_truth_triplets_by_min[min_pairs] = triplets

    # Vẽ ROC và Precision-Recall curves
    plt.figure(figsize=(12, 5))
    
    # Vẽ ROC curves
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Vẽ Precision-Recall curves
    plt.subplot(1, 2, 2)
    
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
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color=colors[min_pairs-1], lw=2, 
                 label=f'Pairs (min={min_pairs}, AUC={roc_auc:.2f})')
        
        # Vẽ Precision-Recall curve
        plt.subplot(1, 2, 2)
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
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color=colors[min_triplets+4], lw=2, linestyle='--', 
                 label=f'Triplets (min={min_triplets}, AUC={roc_auc:.2f})')
        
        # Vẽ Precision-Recall curve
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color=colors[min_triplets+4], lw=2, linestyle='--', 
                 label=f'Triplets (min={min_triplets}, AP={average_precision:.2f})')
    
    # Hoàn thiện biểu đồ ROC
    plt.subplot(1, 2, 1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize=8)
    
    # Hoàn thiện biểu đồ Precision-Recall
    plt.subplot(1, 2, 2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower left", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return {
        'pairs_metrics': {i: calculate_roc_pr_metrics(model_predictions, ground_truth_pairs_by_min[i]) for i in range(1, 6)},
        'triplets_metrics': {i: calculate_roc_pr_metrics(model_predictions, ground_truth_triplets_by_min[i]) for i in range(1, 6)}
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
        pairs = query_images_by_pairs_parallel([predictions], min_pairs)
        triplets = query_images_triplets_parallel([predictions], min_pairs)
        ground_truth_pairs.extend(pairs)
        ground_truth_triplets.extend(triplets)

    # Đánh giá mô hình với dữ liệu pairs
    pairs_metrics = evaluate_model_with_data(model_predictions, ground_truth_pairs, save_plots=False)
    
    # Đánh giá mô hình với dữ liệu triplets
    triplets_metrics = evaluate_model_with_data(model_predictions, ground_truth_triplets, save_plots=False)
    
    # Hàm helper để chuyển đổi dữ liệu thành định dạng JSON-safe
    def convert_to_json_safe(data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        elif isinstance(data, list):
            return [convert_to_json_safe(item) for item in data]
        elif isinstance(data, dict):
            return {key: convert_to_json_safe(value) for key, value in data.items()}
        else:
            return data
    
    # Chuẩn bị kết quả để lưu vào JSON
    results = {
        'image_path': image_path,
        'pairs_metrics': {
            'precision': convert_to_json_safe(pairs_metrics['precision']),
            'recall': convert_to_json_safe(pairs_metrics['recall']),
            'f1': convert_to_json_safe(pairs_metrics['f1']),
            'y_true': convert_to_json_safe(pairs_metrics['y_true']),
            'y_score': convert_to_json_safe(pairs_metrics['y_score'])
        },
        'triplets_metrics': {
            'precision': convert_to_json_safe(triplets_metrics['precision']),
            'recall': convert_to_json_safe(triplets_metrics['recall']),
            'f1': convert_to_json_safe(triplets_metrics['f1']),
            'y_true': convert_to_json_safe(triplets_metrics['y_true']),
            'y_score': convert_to_json_safe(triplets_metrics['y_score'])
        }
    }
    
    # Lưu kết quả vào file JSON nếu cần
    if save_results:
        with open("evaluation_metrics.json", "w") as f:
            json.dump(results, f, indent=4)
    
    return results

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

def evaluate_model_batch(image_paths, model_path, min_pairs_range=(1, 6), max_workers=5, save_results=True):
    """
    Đánh giá mô hình trên nhiều ảnh cùng lúc
    
    Args:
        image_paths: Danh sách đường dẫn đến các ảnh cần đánh giá
        model_path: Đường dẫn đến file checkpoint của mô hình
        min_pairs_range: Tuple (start, end) cho range của min_pairs
        max_workers: Số lượng worker tối đa cho việc xử lý song song
        save_results: Nếu True, lưu kết quả vào file JSON
        
    Returns:
        all_results: Dictionary chứa kết quả đánh giá cho tất cả các ảnh
    """
    print(f"Đánh giá mô hình trên {len(image_paths)} ảnh...")
    
    # Tạo thư mục kết quả nếu chưa tồn tại
    results_dir = "./evaluation_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Đã tạo thư mục kết quả: {results_dir}")
    
    # Khởi tạo danh sách kết quả
    all_results = []
    
    # Hàm worker để xử lý một ảnh
    def process_image(image_path):
        try:
            # Tải mô hình và thực hiện dự đoán
            predictions = load_model_and_predict(image_path, model_path)
            model_predictions = get_predictions_from_model(predictions)
            
            # Khởi tạo danh sách trống cho ground truth
            ground_truth_pairs = []
            ground_truth_triplets = []
            
            # Dùng vòng lặp để truy vấn các pairs và triplets
            for min_pairs in range(min_pairs_range[0], min_pairs_range[1]):
                pairs = query_images_by_pairs_parallel([predictions], min_pairs, max_workers=1)
                triplets = query_images_triplets_parallel([predictions], min_pairs, max_workers=1)
                ground_truth_pairs.extend(pairs)
                ground_truth_triplets.extend(triplets)
            
            # Đánh giá mô hình với dữ liệu pairs
            pairs_metrics = evaluate_model_with_data(model_predictions, ground_truth_pairs, save_plots=False)
            
            # Đánh giá mô hình với dữ liệu triplets
            triplets_metrics = evaluate_model_with_data(model_predictions, ground_truth_triplets, save_plots=False)
            
            # Chuẩn bị kết quả
            result = {
                'image_path': image_path,
                'pairs_metrics': {
                    'precision': pairs_metrics['precision'],
                    'recall': pairs_metrics['recall'],
                    'f1': pairs_metrics['f1'],
                    'y_true': pairs_metrics['y_true'],
                    'y_score': pairs_metrics['y_score']
                },
                'triplets_metrics': {
                    'precision': triplets_metrics['precision'],
                    'recall': triplets_metrics['recall'],
                    'f1': triplets_metrics['f1'],
                    'y_true': triplets_metrics['y_true'],
                    'y_score': triplets_metrics['y_score']
                }
            }
            
            return result
        except Exception as e:
            print(f"Lỗi khi đánh giá ảnh {image_path}: {str(e)}")
            return None
    
    # Sử dụng ThreadPoolExecutor để xử lý song song
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Gửi tất cả các ảnh để xử lý
        future_to_image = {executor.submit(process_image, image_path): image_path for image_path in image_paths}
        
        # Thu thập kết quả khi hoàn thành
        for future in concurrent.futures.as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                    print(f"Đã đánh giá xong ảnh: {image_path}")
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
    
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
    
    # Lưu kết quả chi tiết vào file JSON
    json_result = {
        'average_metrics': avg_metrics,
        'individual_results': []
    }
    
    # Thêm kết quả từng ảnh vào JSON
    for result in all_results:
        json_result['individual_results'].append({
            'image_path': result['image_path'],
            'pairs_metrics': {
                'precision': result['pairs_metrics']['precision'],
                'recall': result['pairs_metrics']['recall'],
                'f1': result['pairs_metrics']['f1']
            },
            'triplets_metrics': {
                'precision': result['triplets_metrics']['precision'],
                'recall': result['triplets_metrics']['recall'],
                'f1': result['triplets_metrics']['f1']
            }
        })
    
    # Lưu kết quả vào file JSON
    if save_results:
        with open(os.path.join(results_dir, "evaluation_metrics_batch.json"), "w") as f:
            json.dump(json_result, f, indent=4)
    
    # In kết quả trung bình
    print("\nKết quả trung bình trên toàn bộ bộ dữ liệu:")
    print("\nMetrics cho pairs:")
    print(f"Precision: {avg_metrics['pairs']['precision']:.4f}, "
          f"Recall: {avg_metrics['pairs']['recall']:.4f}, "
          f"F1: {avg_metrics['pairs']['f1']:.4f}")
    
    print("\nMetrics cho triplets:")
    print(f"Precision: {avg_metrics['triplets']['precision']:.4f}, "
          f"Recall: {avg_metrics['triplets']['recall']:.4f}, "
          f"F1: {avg_metrics['triplets']['f1']:.4f}")
    
    if save_results:
        print(f"\nKết quả chi tiết đã được lưu vào file '{os.path.join(results_dir, 'evaluation_metrics_batch.json')}'")
    
    return all_results

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
    # Sử dụng hàm song song với một ảnh duy nhất
    return query_images_by_pairs_parallel([predictions], min_pairs, max_workers=1, image_folder=image_folder)

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
    # Sử dụng hàm song song với một ảnh duy nhất
    return query_images_triplets_parallel([predictions], min_pairs, max_workers=1, image_folder=image_folder)

def plot_all_data_curves(all_results, save_path=None):
    """
    Vẽ tất cả dữ liệu từ các ảnh trên hai biểu đồ riêng biệt
    
    Args:
        all_results: Danh sách kết quả đánh giá của tất cả ảnh
        save_path: Đường dẫn để lưu đồ thị (nếu None thì hiển thị đồ thị)
    """
    if not all_results:
        print("No results to plot")
        return
        
    # Màu sắc cho các đường cong
    colors = ['red', 'blue', 'green', 'purple', 'orange', 
              'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Vẽ biểu đồ ROC
    plt.figure(figsize=(10, 8))
    
    # Vẽ đường ROC cho từng min_pairs từ 1-5
    for min_pairs in range(1, 6):
        # Tính toán metrics cho pairs
        pairs_y_true = []
        pairs_y_score = []
        for result in all_results:
            if result['pairs_metrics']['y_true'] and result['pairs_metrics']['y_score']:
                pairs_y_true.extend(result['pairs_metrics']['y_true'])
                pairs_y_score.extend(result['pairs_metrics']['y_score'])
        
        if pairs_y_true and pairs_y_score:
            pairs_y_true = np.array(pairs_y_true)
            pairs_y_score = np.array(pairs_y_score)
            
            # Đảm bảo dữ liệu có giá trị 0 và 1
            if len(np.unique(pairs_y_true)) >= 2:
                fpr, tpr, _ = roc_curve(pairs_y_true, pairs_y_score, pos_label=1)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[min_pairs-1], lw=2, 
                        label=f'Images Pairs (min={min_pairs}, AUC={roc_auc:.2f})')
        
        # Tính toán metrics cho triplets
        triplets_y_true = []
        triplets_y_score = []
        for result in all_results:
            if result['triplets_metrics']['y_true'] and result['triplets_metrics']['y_score']:
                triplets_y_true.extend(result['triplets_metrics']['y_true'])
                triplets_y_score.extend(result['triplets_metrics']['y_score'])
        
        if triplets_y_true and triplets_y_score:
            triplets_y_true = np.array(triplets_y_true)
            triplets_y_score = np.array(triplets_y_score)
            
            # Đảm bảo dữ liệu có giá trị 0 và 1
            if len(np.unique(triplets_y_true)) >= 2:
                fpr, tpr, _ = roc_curve(triplets_y_true, triplets_y_score, pos_label=1)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[min_pairs+4], lw=2, linestyle='--',
                        label=f'Images Triplets (min={min_pairs}, AUC={roc_auc:.2f})')
    
    # Vẽ đường chéo ngẫu nhiên
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize=8)
    
    # Lưu biểu đồ ROC
    if save_path:
        roc_path = save_path.replace('.png', '_roc.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ ROC vào {roc_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Vẽ biểu đồ Precision-Recall
    plt.figure(figsize=(10, 8))
    
    # Vẽ đường Precision-Recall cho từng min_pairs từ 1-5
    for min_pairs in range(1, 6):
        # Tính toán metrics cho pairs
        pairs_y_true = []
        pairs_y_score = []
        for result in all_results:
            if result['pairs_metrics']['y_true'] and result['pairs_metrics']['y_score']:
                pairs_y_true.extend(result['pairs_metrics']['y_true'])
                pairs_y_score.extend(result['pairs_metrics']['y_score'])
        
        if pairs_y_true and pairs_y_score:
            pairs_y_true = np.array(pairs_y_true)
            pairs_y_score = np.array(pairs_y_score)
            
            # Đảm bảo dữ liệu có giá trị 0 và 1
            if len(np.unique(pairs_y_true)) >= 2:
                precision, recall, _ = precision_recall_curve(pairs_y_true, pairs_y_score, pos_label=1)
                average_precision = average_precision_score(pairs_y_true, pairs_y_score)
                plt.plot(recall, precision, color=colors[min_pairs-1], lw=2,
                        label=f'Images Pairs (min={min_pairs}, AP={average_precision:.2f})')
        
        # Tính toán metrics cho triplets
        triplets_y_true = []
        triplets_y_score = []
        for result in all_results:
            if result['triplets_metrics']['y_true'] and result['triplets_metrics']['y_score']:
                triplets_y_true.extend(result['triplets_metrics']['y_true'])
                triplets_y_score.extend(result['triplets_metrics']['y_score'])
        
        if triplets_y_true and triplets_y_score:
            triplets_y_true = np.array(triplets_y_true)
            triplets_y_score = np.array(triplets_y_score)
            
            # Đảm bảo dữ liệu có giá trị 0 và 1
            if len(np.unique(triplets_y_true)) >= 2:
                precision, recall, _ = precision_recall_curve(triplets_y_true, triplets_y_score, pos_label=1)
                average_precision = average_precision_score(triplets_y_true, triplets_y_score)
                plt.plot(recall, precision, color=colors[min_pairs+4], lw=2, linestyle='--',
                        label=f'Images Triplets (min={min_pairs}, AP={average_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left", fontsize=8)
    
    # Lưu biểu đồ Precision-Recall
    if save_path:
        pr_path = save_path.replace('.png', '_pr.png')
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ Precision-Recall vào {pr_path}")
    else:
        plt.show()
    
    plt.close()

def evaluate_model_on_dataset(image_folder, model_path, min_pairs_range=(1, 6), save_results=True, max_images=None, max_workers=5):
    """
    Đánh giá mô hình trên toàn bộ bộ dữ liệu ảnh
    
    Args:
        image_folder: Thư mục chứa ảnh cần đánh giá
        model_path: Đường dẫn đến file checkpoint của mô hình
        min_pairs_range: Tuple (start, end) cho range của min_pairs
        save_results: Nếu True, lưu kết quả vào file JSON
        max_images: Số lượng ảnh tối đa cần đánh giá (None nếu đánh giá tất cả)
        max_workers: Số lượng worker tối đa cho việc xử lý song song
        
    Returns:
        results: Dictionary chứa kết quả đánh giá
    """
    # Lấy danh sách tất cả các ảnh trong thư mục
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Giới hạn số lượng ảnh nếu cần
    if max_images is not None and max_images < len(image_files):
        image_files = image_files[:max_images]
    
    # Tạo danh sách đường dẫn đầy đủ
    image_paths = [os.path.join(image_folder, image_file) for image_file in image_files]
    
    # Sử dụng hàm đánh giá hàng loạt
    return evaluate_model_batch(image_paths, model_path, min_pairs_range, max_workers, save_results)

# Thực thi đánh giá mô hình
if __name__ == "__main__":
    # Đường dẫn mặc định
    model_path = './RelTR/ckpt/fine_tune1/checkpoint0049.pth'
    image_folder = "./image_test"  # Thư mục chứa các ảnh cần đánh giá
    
    # Tạo thư mục kết quả nếu chưa tồn tại
    results_dir = "./evaluation_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Đã tạo thư mục kết quả: {results_dir}")
    
    # Lấy danh sách tất cả các ảnh trong thư mục
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    print(f"Tìm thấy {len(image_paths)} ảnh trong thư mục {image_folder}")
    
    # Đánh giá mô hình theo batch
    batch_size = 4  # Có thể điều chỉnh tùy theo GPU memory
    all_results = evaluate_model_batch(image_paths, model_path, batch_size)
    
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
    
    # Vẽ biểu đồ tổng hợp cho tất cả dữ liệu
    print("\nVẽ biểu đồ tổng hợp cho tất cả dữ liệu...")
    plot_all_data_curves(
        all_results,
        save_path=os.path.join(results_dir, "all_data_curves.png")
    )
    
    # Lưu kết quả chi tiết vào file JSON
    json_result = {
        'average_metrics': avg_metrics,
        'individual_results': []
    }
    
    # Thêm kết quả từng ảnh vào JSON
    for result in all_results:
        json_result['individual_results'].append({
            'image_file': result['image_file'],
            'pairs_metrics': {
                'precision': result['pairs_metrics']['precision'],
                'recall': result['pairs_metrics']['recall'],
                'f1': result['pairs_metrics']['f1']
            },
            'triplets_metrics': {
                'precision': result['triplets_metrics']['precision'],
                'recall': result['triplets_metrics']['recall'],
                'f1': result['triplets_metrics']['f1']
            }
        })
    
    # Lưu kết quả vào file JSON
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(json_result, f, indent=4)
    
    # In kết quả trung bình
    print("\nKết quả trung bình trên toàn bộ bộ dữ liệu:")
    print("\nMetrics cho pairs:")
    print(f"Precision: {avg_metrics['pairs']['precision']:.4f}, "
          f"Recall: {avg_metrics['pairs']['recall']:.4f}, "
          f"F1: {avg_metrics['pairs']['f1']:.4f}")
    
    print("\nMetrics cho triplets:")
    print(f"Precision: {avg_metrics['triplets']['precision']:.4f}, "
          f"Recall: {avg_metrics['triplets']['recall']:.4f}, "
          f"F1: {avg_metrics['triplets']['f1']:.4f}")
    
    print(f"\nKết quả chi tiết đã được lưu vào file '{os.path.join(results_dir, 'evaluation_metrics.json')}'")
    print(f"Biểu đồ tổng hợp đã được lưu vào file '{os.path.join(results_dir, 'all_data_curves.png')}'")