from neo4j import GraphDatabase
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def connect_to_neo4j():
    """Kết nối đến Neo4j database"""
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "12345678"
    return GraphDatabase.driver(uri, auth=(username, password))

def get_subject_object_statistics():
    """Lấy thống kê về tần suất xuất hiện của subjects và objects"""
    driver = connect_to_neo4j()
    
    # Khởi tạo counters
    subject_counter = Counter()
    object_counter = Counter()
    
    try:
        with driver.session() as session:
            # Query để đếm subjects
            subject_query = """
            MATCH (s:Object)
            RETURN s.category as category, COUNT(*) as count
            ORDER BY count DESC
            """
            
            # Query để đếm objects
            object_query = """
            MATCH (o:Object)
            RETURN o.category as category, COUNT(*) as count
            ORDER BY count DESC
            """
            
            # Thực hiện queries
            subject_results = session.run(subject_query)
            object_results = session.run(object_query)
            
            # Xử lý kết quả
            for record in subject_results:
                subject_counter[record['category']] = record['count']
                
            for record in object_results:
                object_counter[record['category']] = record['count']
                
    finally:
        driver.close()
    
    return subject_counter, object_counter

def get_relationship_statistics():
    """Lấy thống kê về tần suất xuất hiện của các mối quan hệ"""
    driver = connect_to_neo4j()
    relationship_counter = Counter()
    
    try:
        with driver.session() as session:
            # Query để đếm các mối quan hệ
            relationship_query = """
            MATCH (s:Object)-[r:RELATIONSHIP]->(o:Object)
            RETURN s.category as subject, r.type as relation, o.category as object, COUNT(*) as count
            ORDER BY count DESC
            """
            
            # Thực hiện query
            results = session.run(relationship_query)
            
            # Xử lý kết quả
            for record in results:
                # Tạo key cho mối quan hệ
                rel_key = f"{record['subject']} -[{record['relation']}]-> {record['object']}"
                relationship_counter[rel_key] = record['count']
                
    finally:
        driver.close()
    
    return relationship_counter

def plot_top_categories(counter, title, top_n=30):
    """Vẽ biểu đồ cho top N categories"""
    # Lấy top N categories
    top_categories = dict(counter.most_common(top_n))
    
    # Tạo DataFrame
    df = pd.DataFrame(list(top_categories.items()), columns=['Category', 'Count'])
    
    # Vẽ biểu đồ
    plt.figure(figsize=(15, 8))
    sns.barplot(data=df, x='Count', y='Category')
    plt.title(title)
    plt.xlabel('Count')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.show()

def plot_top_relationships(relationship_counter, top_n=30):
    """Vẽ biểu đồ cho top N relationships"""
    # Lấy top N relationships
    top_relationships = dict(relationship_counter.most_common(top_n))
    
    # Tạo DataFrame
    df = pd.DataFrame(list(top_relationships.items()), columns=['Relationship', 'Count'])
    
    # Vẽ biểu đồ
    plt.figure(figsize=(15, 10))
    sns.barplot(data=df, x='Count', y='Relationship')
    plt.title(f"Top {top_n} Most Common Relationships")
    plt.xlabel('Count')
    plt.ylabel('Relationship')
    plt.tight_layout()
    plt.show()

def save_statistics_to_csv(subject_counter, object_counter, relationship_counter):
    """Lưu thống kê vào file CSV"""
    # Tạo DataFrame cho subjects
    subject_df = pd.DataFrame(list(subject_counter.items()), 
                            columns=['Category', 'Count'])
    subject_df['Type'] = 'Subject'
    
    # Tạo DataFrame cho objects
    object_df = pd.DataFrame(list(object_counter.items()), 
                           columns=['Category', 'Count'])
    object_df['Type'] = 'Object'
    
    # Tạo DataFrame cho relationships
    relationship_df = pd.DataFrame(list(relationship_counter.items()),
                                 columns=['Relationship', 'Count'])
    relationship_df['Type'] = 'Relationship'
    
    # Kết hợp các DataFrame
    combined_df = pd.concat([subject_df, object_df, relationship_df])
    
    # Sắp xếp theo Count giảm dần
    combined_df = combined_df.sort_values('Count', ascending=False)
    
    # Lưu vào CSV
    combined_df.to_csv('category_statistics.csv', index=False)
    print("Statistics saved to category_statistics.csv")

def main():
    # Lấy thống kê
    subject_counter, object_counter = get_subject_object_statistics()
    relationship_counter = get_relationship_statistics()
    
    # In thống kê
    print("\nTop 20 Subjects:")
    for category, count in subject_counter.most_common(20):
        print(f"{category}: {count}")
        
    print("\nTop 20 Objects:")
    for category, count in object_counter.most_common(20):
        print(f"{category}: {count}")
    
    print("\nTop 20 Relationships:")
    for rel, count in relationship_counter.most_common(20):
        print(f"{rel}: {count}")
    
    # Vẽ biểu đồ
    plot_top_categories(subject_counter, "Top 30 Most Frequent Subjects")
    plot_top_categories(object_counter, "Top 30 Most Frequent Objects")
    plot_top_relationships(relationship_counter)
    
    # Lưu vào CSV
    save_statistics_to_csv(subject_counter, object_counter, relationship_counter)
    
    # In tổng số lượng unique categories
    print(f"\nTotal unique subjects: {len(subject_counter)}")
    print(f"Total unique objects: {len(object_counter)}")
    print(f"Total unique relationships: {len(relationship_counter)}")

if __name__ == "__main__":
    main() 