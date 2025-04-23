import json

# Đọc file JSON
with open("./data/vg_focused/test.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Xóa dữ liệu trùng trong images
unique_images = {img["id"]: img for img in data["images"]}.values()

# Xóa dữ liệu trùng trong annotations
unique_annotations = {ann["id"]: ann for ann in data["annotations"]}.values()

# Cập nhật lại dữ liệu
data["images"] = list(unique_images)
data["annotations"] = list(unique_annotations)

# Ghi lại vào file JSON mới (hoặc ghi đè)
with open("./data/vg_focused/test.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)

print("✅ Đã xóa dữ liệu trùng lặp và lưu vào 'cleaned_data.json'")