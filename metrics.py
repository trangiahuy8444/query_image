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
    y_true = []
    y_scores = []
    
    # Nếu không có ground truth, coi tất cả dự đoán là false positives
    if not ground_truth:
        y_true = [0] * len(predictions)
        y_scores = [pred['relation']['score'] for pred in predictions]
    else:
        # Tạo y_true và y_scores với độ dài bằng nhau
        for pred in predictions:
            pred_tuple = (pred['subject']['class'], pred['relation']['class'], pred['object']['class'])
            y_true.append(1 if pred_tuple in ground_truth_pairs else 0)
            y_scores.append(pred['relation']['score'])
    
    # Chỉ tính AUC-ROC và Average Precision nếu có ít nhất 2 lớp
    if len(set(y_true)) >= 2:
        auc_roc = auc(*roc_curve(y_true, y_scores)[:2])
        avg_precision = average_precision_score(y_true, y_scores)
    else:
        auc_roc = 0.0
        avg_precision = 0.0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc_roc,
        'Average Precision': avg_precision,
        'True Positives': tp,
        'False Positives': fp,
        'False Negatives': fn
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
            print(f"\n{'='*50}")
            print(f"Querying Neo4j for image_id: {image_id}")
            print(f"{'='*50}")
            
            # Kiểm tra xem image_id có tồn tại trong database không
            check_query = """
            MATCH (n:Object)
            WHERE n.image_id = $image_id
            RETURN count(n) as node_count
            """
            print(f"Executing query: {check_query}")
            result = session.run(check_query, {"image_id": image_id})
            node_count = result.single()["node_count"]
            print(f"Number of nodes found for image_id: {node_count}")
            
            if node_count == 0:
                print(f"Warning: No nodes found for image_id {image_id}")
                return []
            
            # Truy vấn mối quan hệ subject-relation-object
            query = """
            MATCH (s:Object)-[r:RELATIONSHIP]->(o:Object)
            WHERE s.image_id = $image_id AND o.image_id = $image_id
            RETURN s.category AS subject, r.type AS relation, o.category AS object
            """
            print(f"Executing query: {query}")
            result = session.run(query, {"image_id": image_id})
            
            # In ra số lượng relationships tìm thấy
            relationships = list(result)
            print(f"Number of relationships found: {len(relationships)}")
            
            # Lưu các mối quan hệ vào ground_truth
            for record in relationships:
                subject = record['subject']
                relation = record['relation']
                object_ = record['object']
                ground_truth.append((subject, relation, object_))
                print(f"Found relationship: {subject} -[{relation}]-> {object_}")

    except Exception as e:
        print(f"Error retrieving ground truth for image {image_id}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

    return ground_truth

def check_neo4j_connection():
    """
    Kiểm tra kết nối đến Neo4j và in ra thông tin cơ bản.
    """
    try:
        with driver.session() as session:
            # Kiểm tra số lượng nodes và relationships
            result = session.run("""
                MATCH (n)
                RETURN 
                    count(n) as total_nodes,
                    count(DISTINCT labels(n)) as label_count,
                    labels(n) as labels
                LIMIT 1
            """)
            
            record = result.single()
            if record:
                print("\nNeo4j Connection Status:")
                print(f"Total nodes: {record['total_nodes']}")
                print(f"Label count: {record['label_count']}")
                print(f"Labels: {record['labels']}")
                
                # Kiểm tra số lượng relationships
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN count(r) as total_relationships
                """)
                rel_count = result.single()['total_relationships']
                print(f"Total relationships: {rel_count}")
                
                return True
            else:
                print("No data found in Neo4j database")
                return False
    except Exception as e:
        print(f"Error connecting to Neo4j: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def check_image_data(image_id):
    """
    Kiểm tra dữ liệu của một ảnh cụ thể trong Neo4j.
    """
    try:
        with driver.session() as session:
            # Kiểm tra nodes của ảnh
            result = session.run("""
                MATCH (n:Object)
                WHERE n.image_id = $image_id
                RETURN count(n) as object_count
            """, {"image_id": image_id})
            
            object_count = result.single()['object_count']
            print(f"\nChecking data for image {image_id}:")
            print(f"Number of objects: {object_count}")
            
            # Kiểm tra relationships của ảnh
            result = session.run("""
                MATCH (s:Object)-[r:RELATIONSHIP]->(o:Object)
                WHERE s.image_id = $image_id AND o.image_id = $image_id
                RETURN count(r) as relationship_count
            """, {"image_id": image_id})
            
            relationship_count = result.single()['relationship_count']
            print(f"Number of relationships: {relationship_count}")
            
            # Lấy danh sách các relationships
            result = session.run("""
                MATCH (s:Object)-[r:RELATIONSHIP]->(o:Object)
                WHERE s.image_id = $image_id AND o.image_id = $image_id
                RETURN s.category as subject, r.type as relation, o.category as object
            """, {"image_id": image_id})
            
            print("\nRelationships found:")
            for record in result:
                print(f"{record['subject']} -[{record['relation']}]-> {record['object']}")
            
            return relationship_count > 0
    except Exception as e:
        print(f"Error checking image data: {str(e)}")
        return False

def evaluate_sample_images(image_ids, model):
    """
    Đánh giá một số ảnh mẫu để kiểm tra kết quả.
    """
    print("\nEvaluating sample images...")
    
    all_predictions = []
    all_ground_truth = []
    all_y_true = []
    all_y_scores = []
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs('./output', exist_ok=True)
    
    # Dictionary để lưu dữ liệu của từng ảnh
    image_data = {}
    
    for image_id in image_ids:
        try:
            print(f"\n{'='*50}")
            print(f"Processing image: {image_id}")
            print(f"{'='*50}")
            
            # Đảm bảo image_id là string và thêm .jpg
            image_id_str = f"{image_id}.jpg" if not image_id.endswith('.jpg') else image_id
            image_path = os.path.join('./image_test', image_id_str)
            
            print(f"Looking for image at path: {image_path}")
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue
            
            # Dự đoán với mô hình
            print("\nMaking predictions with the model...")
            predictions = predict(image_path, model)
            print(f"Number of predictions: {len(predictions)}")
            
            # Format predictions for JSON
            formatted_predictions = []
            if predictions:
                print("Predictions:")
                for i, pred in enumerate(predictions, 1):
                    pred_info = {
                        'subject': pred['subject']['class'],
                        'relation': pred['relation']['class'],
                        'object': pred['object']['class'],
                        'score': float(pred['relation']['score'])
                    }
                    formatted_predictions.append(pred_info)
                    print(f"{i}. {pred_info['subject']} -[{pred_info['relation']}]-> {pred_info['object']} (score: {pred_info['score']:.4f})")
            else:
                print("No predictions made by the model")
            
            # Lấy ground truth từ Neo4j
            print("\nRetrieving ground truth from Neo4j...")
            ground_truth = get_ground_truth_from_neo4j(image_id)
            print(f"Number of ground truth relationships: {len(ground_truth)}")
            
            # Format ground truth for JSON
            formatted_ground_truth = []
            if ground_truth:
                print("Ground truth relationships:")
                for i, gt in enumerate(ground_truth, 1):
                    gt_info = {
                        'subject': gt[0],
                        'relation': gt[1],
                        'object': gt[2]
                    }
                    formatted_ground_truth.append(gt_info)
                    print(f"{i}. {gt_info['subject']} -[{gt_info['relation']}]-> {gt_info['object']}")
            else:
                print("No ground truth relationships found in Neo4j")
            
            # Tính toán metrics
            print("\nCalculating metrics...")
            metrics = calculate_metrics(predictions, ground_truth)
            print("Metrics for this image:")
            print(json.dumps(metrics, indent=4))
            
            # Lưu dữ liệu của ảnh này
            image_data[image_id] = {
                'image_path': image_path,
                'predictions': formatted_predictions,
                'ground_truth': formatted_ground_truth,
                'metrics': metrics
            }
            
            # Lưu dữ liệu để vẽ biểu đồ
            y_true = [1 if (s, r, o) in ground_truth else 0 for (s, r, o) in predictions]
            y_scores = [pred['relation']['score'] for pred in predictions]
            
            all_predictions.extend(predictions)
            all_ground_truth.extend(ground_truth)
            all_y_true.extend(y_true)
            all_y_scores.extend(y_scores)
            
        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    # Lưu dữ liệu của tất cả ảnh vào file JSON
    output_json_path = './output/sample_images_data.json'
    print(f"\nSaving data to {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(image_data, f, indent=4, ensure_ascii=False)
    
    if all_y_true and all_y_scores:
        print("\nPlotting curves for all sample images...")
        # Vẽ biểu đồ tổng hợp
        plt.figure(figsize=(12, 6))
        
        # Vẽ ROC curve
        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Vẽ Precision-Recall curve
        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)
        avg_precision = average_precision_score(all_y_true, all_y_scores)
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        plt.tight_layout()
        output_path = './output/sample_images_curves.png'
        print(f"Saving curves to: {output_path}")
        plt.savefig(output_path)
        plt.close()
        
        # Tính toán metrics tổng hợp
        print("\nCalculating total metrics for all sample images...")
        total_metrics = calculate_metrics(all_predictions, all_ground_truth)
        print("Total metrics:")
        print(json.dumps(total_metrics, indent=4))
        
        # Lưu metrics tổng hợp vào file JSON
        total_metrics_path = './output/total_metrics.json'
        print(f"Saving total metrics to {total_metrics_path}")
        with open(total_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(total_metrics, f, indent=4, ensure_ascii=False)
    else:
        print("\nNo valid data to plot curves or calculate total metrics")

def categorize_images_by_pairs(min_pairs=1, max_pairs=5):
    """
    Phân loại ảnh theo số lượng cặp subject-object.
    """
    categories = {}
    for num_pairs in range(min_pairs, max_pairs + 1):
        with driver.session() as session:
            query = """
            MATCH (s:Object)-[r:RELATIONSHIP]->(o:Object)
            WITH s.image_id as image_id, count(DISTINCT [s.category, o.category]) as pair_count
            WHERE pair_count >= $num_pairs
            RETURN image_id, pair_count
            ORDER BY pair_count DESC
            """
            result = session.run(query, {"num_pairs": num_pairs})
            image_ids = [str(record["image_id"]) for record in result]  # Chuyển đổi image_id thành string
            categories[f"pairs_{num_pairs}"] = image_ids
    return categories

def categorize_images_by_triplets(min_triplets=1, max_triplets=5):
    """
    Phân loại ảnh theo số lượng triplets (subject-relation-object).
    """
    categories = {}
    for num_triplets in range(min_triplets, max_triplets + 1):
        with driver.session() as session:
            query = """
            MATCH (s:Object)-[r:RELATIONSHIP]->(o:Object)
            WITH s.image_id as image_id, count(DISTINCT [s.category, r.type, o.category]) as triplet_count
            WHERE triplet_count >= $num_triplets
            RETURN image_id, triplet_count
            ORDER BY triplet_count DESC
            """
            result = session.run(query, {"num_triplets": num_triplets})
            image_ids = [str(record["image_id"]) for record in result]  # Chuyển đổi image_id thành string
            categories[f"triplets_{num_triplets}"] = image_ids
    return categories

def evaluate_category(category_name, image_ids, model):
    """
    Đánh giá một danh mục ảnh cụ thể.
    """
    metrics_list = []
    y_true = []
    y_scores = []
    
    print(f"\nEvaluating category: {category_name}")
    print(f"Number of images in category: {len(image_ids)}")
    
    for image_id in image_ids:
        try:
            # Đảm bảo image_id là string và thêm .jpg
            image_id_str = f"{image_id}.jpg" if not image_id.endswith('.jpg') else image_id
            image_path = os.path.join('./data/vg_focused/images', image_id_str)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue
                
            predictions = predict(image_path, model)
            ground_truth = get_ground_truth_from_neo4j(image_id)
            
            metrics = calculate_metrics(predictions, ground_truth)
            metrics_list.append(metrics)
            
            # Lưu các giá trị thực tế và dự đoán
            y_true.extend([1 if (s, r, o) in ground_truth else 0 for (s, r, o) in predictions])
            y_scores.extend([pred['relation']['score'] for pred in predictions])
            
        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
            continue
    
    if not metrics_list:
        print(f"Warning: No valid metrics for category {category_name}")
        return None, None, None
    
    # Tính toán metrics trung bình
    avg_metrics = {
        'Precision': np.mean([m['Precision'] for m in metrics_list]),
        'Recall': np.mean([m['Recall'] for m in metrics_list]),
        'F1 Score': np.mean([m['F1 Score'] for m in metrics_list]),
        'AUC-ROC': np.mean([m['AUC-ROC'] for m in metrics_list]),
        'Average Precision': np.mean([m['Average Precision'] for m in metrics_list])
    }
    
    return avg_metrics, y_true, y_scores

def plot_category_metrics(categories_metrics, output_dir='./output'):
    """
    Vẽ biểu đồ so sánh metrics giữa các danh mục.
    """
    categories = list(categories_metrics.keys())
    metrics = ['Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'Average Precision']
    
    x = np.arange(len(categories))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for i, metric in enumerate(metrics):
        values = [categories_metrics[cat][metric] for cat in categories]
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_xlabel('Categories')
    ax.set_ylabel('Score')
    ax.set_title('Metrics Comparison Across Categories')
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(categories, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_metrics_comparison.png'))
    plt.close()

def plot_all_curves(categories_data, output_dir='./output'):
    """
    Vẽ tất cả các đường ROC và Precision-Recall cho 10 danh mục trên cùng một biểu đồ.
    """
    # Chuẩn bị màu sắc cho các đường
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Vẽ ROC curves
    plt.figure(figsize=(12, 6))
    for i, (category, (y_true, y_scores)) in enumerate(categories_data.items()):
        if len(set(y_true)) >= 2:  # Chỉ vẽ nếu có ít nhất 2 lớp
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{category} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Categories')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'all_categories_roc_curves.png'))
    plt.close()
    
    # Vẽ Precision-Recall curves
    plt.figure(figsize=(12, 6))
    for i, (category, (y_true, y_scores)) in enumerate(categories_data.items()):
        if len(set(y_true)) >= 2:  # Chỉ vẽ nếu có ít nhất 2 lớp
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            avg_precision = average_precision_score(y_true, y_scores)
            plt.plot(recall, precision, color=colors[i], lw=2,
                    label=f'{category} (AP = {avg_precision:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for All Categories')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'all_categories_precision_recall_curves.png'))
    plt.close()

def evaluate_all_categories(output_dir='./output'):
    """
    Đánh giá tất cả các danh mục và vẽ biểu đồ.
    """
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mô hình
    model = load_model('./RelTR/ckpt/fine_tune1/checkpoint0049.pth')
    
    # Phân loại ảnh
    print("\nCategorizing images by pairs...")
    pair_categories = categorize_images_by_pairs()
    print("\nCategorizing images by triplets...")
    triplet_categories = categorize_images_by_triplets()
    
    # Đánh giá từng danh mục
    categories_metrics = {}
    categories_data = {}  # Lưu trữ dữ liệu để vẽ curves
    
    # Đánh giá các danh mục pairs
    print("\nEvaluating pair categories...")
    for category, image_ids in pair_categories.items():
        if image_ids:  # Chỉ đánh giá nếu có ảnh trong danh mục
            print(f"\nProcessing category: {category}")
            result = evaluate_category(category, image_ids, model)
            if result is not None:
                metrics, y_true, y_scores = result
                categories_metrics[category] = metrics
                categories_data[category] = (y_true, y_scores)
    
    # Đánh giá các danh mục triplets
    print("\nEvaluating triplet categories...")
    for category, image_ids in triplet_categories.items():
        if image_ids:  # Chỉ đánh giá nếu có ảnh trong danh mục
            print(f"\nProcessing category: {category}")
            result = evaluate_category(category, image_ids, model)
            if result is not None:
                metrics, y_true, y_scores = result
                categories_metrics[category] = metrics
                categories_data[category] = (y_true, y_scores)
    
    if not categories_metrics:
        print("\nWarning: No valid metrics were calculated for any category")
        return None
    
    # Vẽ tất cả các curves trên cùng một biểu đồ
    plot_all_curves(categories_data, output_dir)
    
    # Lưu kết quả vào file JSON
    with open(os.path.join(output_dir, 'category_metrics.json'), 'w') as f:
        json.dump(categories_metrics, f, indent=4)
    
    return categories_metrics

if __name__ == "__main__":
    # Kiểm tra kết nối Neo4j trước
    print("Checking Neo4j connection...")
    if not check_neo4j_connection():
        print("Cannot proceed with testing due to Neo4j connection issues")
        exit(1)
    
    # Load mô hình
    print("Loading model...")
    model = load_model('./RelTR/ckpt/fine_tune1/checkpoint0049.pth')
    print("Model loaded successfully")
    
    # Test một số ảnh mẫu
    sample_image_ids = [
        "150542",  # Ảnh có ít nhất 1 cặp
        "286068",  # Ảnh có ít nhất 2 cặp
        "498377"   # Ảnh có ít nhất 3 cặp
    ]
    
    print(f"\nStarting evaluation of {len(sample_image_ids)} sample images...")
    # Đánh giá ảnh mẫu
    evaluate_sample_images(sample_image_ids, model)
    
    # Sau khi kiểm tra xong, bạn có thể chạy đánh giá toàn bộ
    # categories_metrics = evaluate_all_categories()
    # if categories_metrics is not None:
    #     print("\nMetrics for all categories:")
    #     print(json.dumps(categories_metrics, indent=4))
    # else:
    #     print("\nEvaluation failed. Please check the logs for details.")