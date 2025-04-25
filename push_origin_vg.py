from neo4j import GraphDatabase
import json
import os
from dotenv import load_dotenv
from typing import List, Dict
import time
import gc

# Load environment variables
load_dotenv()

# Neo4j connection
uri = os.getenv("NEO4J_URI", "neo4j+s://b40b4f2a.databases.neo4j.io")
username = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "fpKNUXKT-4z0kQMm1nuUaiXe8p70uIebc3y3a4Z8kUA")
driver = GraphDatabase.driver(uri, auth=(username, password))

# Optimized batch sizes for 16GB RAM
OBJECT_BATCH_SIZE = 5000  # Increased for objects
REL_BATCH_SIZE = 2000    # Smaller for relationships due to MATCH operations

def create_objects_batch(tx, objects_batch):
    query = """
    UNWIND $objects as obj
    MERGE (o:Object {id: obj.id})
    SET o.category = obj.category,
        o.bbox = obj.bbox,
        o.area = obj.area,
        o.image_id = obj.image_id
    """
    tx.run(query, objects=objects_batch)

def create_relationships_batch(tx, rels_batch):
    query = """
    UNWIND $rels as rel
    MATCH (s:Object {id: rel.sub_id})
    MATCH (o:Object {id: rel.obj_id})
    MERGE (s)-[:RELATIONSHIP {type: rel.rel_name}]->(o)
    """
    tx.run(query, rels=rels_batch)

def add_all_images_to_neo4j(json_folder):
    json_files = ["train.json", "val.json", "test.json"]
    rel_file = "rel.json"
    
    # Load relationship data
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
    
    # Clear existing data
    with driver.session() as session:
        try:
            session.run("MATCH (n) DETACH DELETE n")
            print("Successfully cleared old data from Neo4j")
        except Exception as e:
            print(f"Error clearing Neo4j database: {e}")
            return

    start_time = time.time()
    
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
        
        # Prepare batches
        objects_batch = []
        relationships_batch = []
        processed_count = 0
        total_objects = 0
        total_relationships = 0
        
        print(f"Processing {file}...")
        
        with driver.session() as session:
            # Process objects in batches
            for image in data['images']:
                try:
                    image_id = image['id']
                    annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
                    
                    # Add objects to batch
                    for ann in annotations:
                        category_name = categories.get(ann['category_id'], f"Category {ann['category_id']}")
                        objects_batch.append({
                            'id': ann['id'],
                            'category': category_name,
                            'bbox': ann['bbox'],
                            'area': ann['area'],
                            'image_id': image_id
                        })
                    
                    # Process relationships
                    relationships = rel_data[file.replace(".json", "")].get(str(image_id), [])
                    for sub_id, obj_id, rel_id in relationships:
                        if sub_id < len(annotations) and obj_id < len(annotations):
                            sub_ann = annotations[sub_id]
                            obj_ann = annotations[obj_id]
                            rel_name = rel_data['rel_categories'][rel_id]
                            
                            relationships_batch.append({
                                'sub_id': sub_ann['id'],
                                'obj_id': obj_ann['id'],
                                'rel_name': rel_name
                            })
                    
                    processed_count += 1
                    
                    # Execute batch when size threshold is reached
                    if len(objects_batch) >= OBJECT_BATCH_SIZE:
                        session.execute_write(create_objects_batch, objects_batch)
                        total_objects += len(objects_batch)
                        objects_batch = []
                        print(f"Processed {processed_count} images, {total_objects} objects from {file}")
                    
                    if len(relationships_batch) >= REL_BATCH_SIZE:
                        session.execute_write(create_relationships_batch, relationships_batch)
                        total_relationships += len(relationships_batch)
                        relationships_batch = []
                        print(f"Processed {processed_count} images, {total_relationships} relationships from {file}")
                
                except Exception as e:
                    print(f"Error processing image {image_id} in {file}: {e}")
                    continue
            
            # Process remaining items in batches
            if objects_batch:
                session.execute_write(create_objects_batch, objects_batch)
                total_objects += len(objects_batch)
            if relationships_batch:
                session.execute_write(create_relationships_batch, relationships_batch)
                total_relationships += len(relationships_batch)
            
            # Clear memory
            objects_batch = None
            relationships_batch = None
            gc.collect()
            
            end_time = time.time()
            print(f"Completed processing {file}")
            print(f"Total images processed: {processed_count}")
            print(f"Total objects created: {total_objects}")
            print(f"Total relationships created: {total_relationships}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    json_folder = os.path.join(project_root, "data/vg_focused")
    
    if not os.path.exists(json_folder):
        print(f"Error: Folder {json_folder} not found!")
    else:
        print(f"Starting to process data from {json_folder}")
        add_all_images_to_neo4j(json_folder)
        print("Process completed!")