from neo4j import GraphDatabase
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "12345678")
driver = GraphDatabase.driver(uri, auth=(username, password))

# Hàm để thêm toàn bộ ảnh từ train, val, test vào Neo4j
def add_all_images_to_neo4j(json_folder):
    json_files = ["train.json", "val.json", "test.json"]
    rel_file = "rel.json"
    
    # Đọc quan hệ từ rel.json
    rel_path = os.path.join(json_folder, rel_file)
    if not os.path.exists(rel_path):
        print(f"Error: {rel_path} not found!")
        return
        
    try:
        with open(rel_path, 'r', encoding='utf-8') as f:
            rel_data = json.load(f)
    except Exception as e:
        print(f"Error reading {rel_path}: {e}")
        return
    
    with driver.session() as session:
        try:
            session.run("MATCH (n) DETACH DELETE n")  # Xóa dữ liệu cũ
            print("Successfully cleared old data from Neo4j")
        except Exception as e:
            print(f"Error clearing Neo4j database: {e}")
            return
    
    for file in json_files:
        file_path = os.path.join(json_folder, file)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        with driver.session() as session:
            print(f"Processing {file}...")
            processed_count = 0
            
            for image in data['images']:
                try:
                    image_id = image['id']
                    annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
                    relationships = rel_data[file.replace(".json", "")].get(str(image_id), [])
                    
                    for ann in annotations:
                        category_name = categories.get(ann['category_id'], f"Category {ann['category_id']}")
                        session.run(
                            """
                            MERGE (o:Object {id: $id, category: $category, bbox: $bbox, area: $area, image_id: $image_id})
                            """,
                            id=ann['id'],
                            category=category_name,
                            bbox=ann['bbox'],
                            area=ann['area'],
                            image_id=image_id,
                        )
                    
                    for sub_id, obj_id, rel_id in relationships:
                        if sub_id < len(annotations) and obj_id < len(annotations):
                            sub_ann = annotations[sub_id]
                            obj_ann = annotations[obj_id]
                            rel_name = rel_data['rel_categories'][rel_id]
                            
                            session.run(
                                """
                                MATCH (s:Object {id: $sub_id}), (o:Object {id: $obj_id})
                                MERGE (s)-[:RELATIONSHIP {type: $rel_name}]->(o)
                                """,
                                sub_id=sub_ann['id'],
                                obj_id=obj_ann['id'],
                                rel_name=rel_name,
                            )
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} images from {file}")
                        
                except Exception as e:
                    print(f"Error processing image {image_id} in {file}: {e}")
                    continue
            
            print(f"Completed processing {file}. Total images processed: {processed_count}")

if __name__ == "__main__":
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    json_folder = os.path.join(project_root, "vg_1000_focused")
    
    if not os.path.exists(json_folder):
        print(f"Error: Folder {json_folder} not found!")
    else:
        print(f"Starting to process data from {json_folder}")
        add_all_images_to_neo4j(json_folder)
        print("Process completed!")