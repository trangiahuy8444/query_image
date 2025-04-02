import json
import os

# Danh sách image_id cần kiểm tra
image_ids = [286068]

# Chuyển đổi image_ids thành set để tìm kiếm nhanh hơn
image_ids_set = set(image_ids)

def examine_json_structure(file_path):
    """Kiểm tra cấu trúc của file JSON"""
    if not os.path.exists(file_path):
        print(f"File {file_path} không tồn tại!")
        return
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"\nCấu trúc của file {file_path}:")
            print(f"Kiểu dữ liệu: {type(data)}")
            if isinstance(data, dict):
                print("\nCác trường có trong file:")
                for key in data.keys():
                    print(f"- {key}")
                if 'images' in data and len(data['images']) > 0:
                    print("\nCấu trúc của phần tử đầu tiên trong images:")
                    print(json.dumps(data['images'][0], indent=2))
            else:
                print("File không phải là dictionary")
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {str(e)}")

def check_image_ids_in_json(file_path):
    """Kiểm tra image_ids trong file JSON"""
    if not os.path.exists(file_path):
        print(f"File {file_path} không tồn tại!")
        return set()
    
    found_ids = set()
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # Xử lý rel.json
            if file_path.endswith('rel.json'):
                # Trong rel.json, image_id là key của dictionary
                for image_id in data.keys():
                    if image_id in image_ids_set:
                        found_ids.add(image_id)
            else:
                # Xử lý train.json, val.json, test.json
                if 'images' in data:
                    for image in data['images']:
                        if 'id' in image and image['id'] in image_ids_set:
                            found_ids.add(image['id'])
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {str(e)}")
    
    return found_ids

# Kiểm tra cấu trúc của từng file trước
files = ['vg_1000/rel.json', 'vg_1000/val.json', 'vg_1000/test.json', 'vg_1000/train.json']
for file in files:
    examine_json_structure(file)

# Sau khi đã xem cấu trúc, kiểm tra image_ids
print("\nBắt đầu kiểm tra image_ids...")
results = {}
for file in files:
    found_ids = check_image_ids_in_json(file)
    results[file] = found_ids
    print(f"\nKết quả trong {file}:")
    print(f"Số lượng image_id tìm thấy: {len(found_ids)}")
    print(f"Các image_id tìm thấy: {sorted(list(found_ids))}")

# Tìm các image_id không xuất hiện trong bất kỳ file nào
all_found_ids = set().union(*results.values())
not_found_ids = image_ids_set - all_found_ids

print("\nTổng kết:")
print(f"Tổng số image_id cần kiểm tra: {len(image_ids)}")
print(f"Tổng số image_id tìm thấy: {len(all_found_ids)}")
print(f"Số lượng image_id không tìm thấy: {len(not_found_ids)}")
if not_found_ids:
    print(f"Các image_id không tìm thấy: {sorted(list(not_found_ids))}") 