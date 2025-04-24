import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from RelTR.inference import load_model, predict  # Import mô hình RelTR
from neo4j import GraphDatabase  # Dùng để truy vấn Neo4j

# Kết nối Neo4j
uri = "bolt://localhost:7689"
username = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(username, password))

def calculate_metrics(predictions, ground_truth):
    """
    Tính toán các chỉ số đánh giá: Precision, Recall, F1-Score, AUC-ROC, Average Precision.
    """
    predicted_pairs = set((pred['subject']['class'], pred['relation']['class'], pred['object']['class']) for pred in predictions)
    ground_truth_pairs = set((gt[0], gt[1], gt[2]) for gt in ground_truth)

    tp = len(predicted_pairs.intersection(ground_truth_pairs))  # True Positives
    fp = len(predicted_pairs - ground_truth_pairs)  # False Positives
    fn = len(ground_truth_pairs - predicted_pairs)  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Tính AUC-ROC và Average Precision
    y_true = [1 if (s, r, o) in ground_truth_pairs else 0 for (s, r, o) in predicted_pairs]
    y_scores = [pred['relation']['score'] for pred in predictions]
    
    auc_roc = auc(*roc_curve(y_true, y_scores)[:2])
    avg_precision = average_precision_score(y_true, y_scores)

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc_roc,
        'Average Precision': avg_precision
    }

def plot_roc_curve(y_true, y_scores, output_path='roc_curve.png'):
    """
    Vẽ ROC Curve và lưu vào file.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random chance
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    return roc_auc

def plot_precision_recall_curve(y_true, y_scores, output_path='precision_recall_curve.png'):
    """
    Vẽ Precision-Recall Curve và lưu vào file.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def query_images_by_pairs_count(predictions, min_pairs):
    """
    Truy vấn các ảnh có ít nhất `min_pairs` cặp subject-object matching.
    """
    image_details = []
    processed_image_ids = set()
    query_start_time = time.time()
    
    with driver.session() as session:
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
        
        total_results = 0
        total_matching_percentage = 0
        
        for record in result:
            image_id = record['image_id']
            if image_id not in processed_image_ids:
                processed_image_ids.add(image_id)
                relationships_str = [f"{rel['subject']} -[{rel['relation']}]-> {rel['object']}" for rel in record['relationships']]
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

def evaluate_and_plot(output_dir='./output', image_folder='./data/vg_focused/images'):
    """
    Đánh giá mô hình trên toàn bộ ảnh trong thư mục và vẽ các biểu đồ.
    """
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Load mô hình RelTR
    model = load_model('./RelTR/ckpt/fine_tune1/checkpoint0049.pth')

    # Lấy danh sách các ảnh
    all_images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    # Dự đoán và tính toán metrics cho tất cả các ảnh
    predictions_all = []
    ground_truth_all = []

    for image_id in all_images:
        image_path = os.path.join(image_folder, image_id)

        # Dự đoán với mô hình
        predictions = predict(image_path, model)

        # Lấy ground truth từ cơ sở dữ liệu Neo4j
        ground_truth = get_ground_truth_from_neo4j(image_id)

        predictions_all.append(predictions)
        ground_truth_all.append(ground_truth)

    # Tính toán và vẽ các biểu đồ cho toàn bộ dữ liệu
    y_true = []
    y_scores = []

    metrics_list = []
    for predictions, ground_truth in zip(predictions_all, ground_truth_all):
        # Tính toán metrics cho từng ảnh
        metrics = calculate_metrics(predictions, ground_truth)
        metrics_list.append(metrics)

        # Lưu các giá trị thực tế và dự đoán
        y_true.extend([1 if (s, r, o) in ground_truth else 0 for (s, r, o) in predictions])
        y_scores.extend([pred['relation']['score'] for pred in predictions])

    # Vẽ các biểu đồ ROC và Precision-Recall
    plot_roc_curve(y_true, y_scores, output_path=os.path.join(output_dir, 'roc_curve.png'))
    plot_precision_recall_curve(y_true, y_scores, output_path=os.path.join(output_dir, 'precision_recall_curve.png'))

    # Tính toán các chỉ số tổng hợp
    avg_metrics = {
        'Precision': np.mean([metrics['Precision'] for metrics in metrics_list]),
        'Recall': np.mean([metrics['Recall'] for metrics in metrics_list]),
        'F1 Score': np.mean([metrics['F1 Score'] for metrics in metrics_list]),
        'AUC-ROC': np.mean([metrics['AUC-ROC'] for metrics in metrics_list]),
        'Average Precision': np.mean([metrics['Average Precision'] for metrics in metrics_list])
    }

    return avg_metrics


def get_ground_truth_from_neo4j(image_id):
    """
    Truy vấn dữ liệu ground truth từ cơ sở dữ liệu Neo4j cho một image_id.

    :param image_id: ID của ảnh (image_id)
    :return: Danh sách các mối quan hệ thực tế dưới dạng [(subject, relation, object)]
    """
    ground_truth = []
    
    try:
        with driver.session() as session:
            # Truy vấn mối quan hệ subject-relation-object trong cơ sở dữ liệu Neo4j
            query = """
            MATCH (s:Object)-[r:RELATIONSHIP]->(o:Object)
            WHERE s.image_id = $image_id AND o.image_id = $image_id
            RETURN s.category AS subject, r.type AS relation, o.category AS object
            """
            result = session.run(query, {"image_id": image_id})

            # Lưu các mối quan hệ vào ground_truth
            for record in result:
                subject = record['subject']
                relation = record['relation']
                object_ = record['object']
                ground_truth.append((subject, relation, object_))

    except Exception as e:
        print(f"Error retrieving ground truth for image {image_id}: {str(e)}")
        # Trường hợp lỗi, có thể trả về một danh sách trống hoặc dữ liệu mặc định
        return []

    return ground_truth

def test_sample_images(image_ids, output_dir='./output'):
    """
    Test đánh giá trên một số ảnh mẫu.
    
    Args:
        image_ids: Danh sách các ID ảnh cần test
        output_dir: Thư mục lưu kết quả
    """
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Load mô hình RelTR
    model = load_model('./RelTR/ckpt/fine_tune1/checkpoint0049.pth')
    
    # Dự đoán và tính toán metrics cho các ảnh mẫu
    predictions_all = []
    ground_truth_all = []
    
    print(f"\nTesting on {len(image_ids)} sample images...")
    
    for image_id in image_ids:
        print(f"\nProcessing image: {image_id}")
        image_path = os.path.join('./data/vg_focused/images', image_id)
        
        # Dự đoán với mô hình
        predictions = predict(image_path, model)
        print(f"Number of predictions: {len(predictions)}")
        
        # Lấy ground truth từ Neo4j
        ground_truth = get_ground_truth_from_neo4j(image_id)
        print(f"Number of ground truth relationships: {len(ground_truth)}")
        
        # Tính toán metrics cho ảnh này
        metrics = calculate_metrics(predictions, ground_truth)
        print("\nMetrics for this image:")
        print(json.dumps(metrics, indent=4))
        
        predictions_all.append(predictions)
        ground_truth_all.append(ground_truth)
    
    # Tính toán và vẽ các biểu đồ cho toàn bộ dữ liệu mẫu
    y_true = []
    y_scores = []
    
    for predictions, ground_truth in zip(predictions_all, ground_truth_all):
        y_true.extend([1 if (s, r, o) in ground_truth else 0 for (s, r, o) in predictions])
        y_scores.extend([pred['relation']['score'] for pred in predictions])
    
    # Vẽ các biểu đồ ROC và Precision-Recall
    plot_roc_curve(y_true, y_scores, output_path=os.path.join(output_dir, 'sample_roc_curve.png'))
    plot_precision_recall_curve(y_true, y_scores, output_path=os.path.join(output_dir, 'sample_precision_recall_curve.png'))
    
    return {
        'predictions': predictions_all,
        'ground_truth': ground_truth_all,
        'y_true': y_true,
        'y_scores': y_scores
    }

if __name__ == "__main__":
    # Test trên một số ảnh mẫu
    sample_image_ids = [
        "1159285.jpg",
        "1159286.jpg",
        "1159287.jpg"
    ]
    
    test_results = test_sample_images(sample_image_ids)
    
    # Sau khi test xong, bạn có thể chạy đánh giá trên toàn bộ dữ liệu
    # metrics = evaluate_and_plot(
    #     output_dir='./output',
    #     image_folder='./data/vg_focused/images'
    # )
    # print("\nMetrics for the entire dataset:")
    # print(json.dumps(metrics, indent=4))