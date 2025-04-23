import json
import os

def delete_image_data(image_id, json_folder, images_folder):
    json_files = ["train.json", "val.json", "test.json", "rel.json"]
    image_found = False

    for file in json_files:
        file_path = os.path.join(json_folder, file)
        # Đọc dữ liệu từ tệp JSON

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if file in ["train.json", "val.json", "test.json"]:
            # Tìm ảnh có `image_id` trong danh sách images
            images_to_delete = [img for img in data.get(
                "images", []) if img["id"] == image_id]
            if images_to_delete:
                image_found = True
                data["images"] = [img for img in data.get(
                    "images", []) if img["id"] != image_id]

                # Xóa annotations liên quan đến image_id
                data["annotations"] = [anno for anno in data.get(
                    "annotations", []) if anno["image_id"] != image_id]

                # Xóa file ảnh trong thư mục images
                for img in images_to_delete:
                    image_path = os.path.join(images_folder, img["file_name"])
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"Đã xóa ảnh: {image_path}")

        elif file == "rel.json":
            # Xóa quan hệ liên quan đến image_id trong train, val, test
            for key in ["train", "val", "test"]:
                if key in data and str(image_id) in data[key]:
                    del data[key][str(image_id)]
                    image_found = True

        # Ghi dữ liệu đã cập nhật trở lại tệp JSON nếu có thay đổi
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    if image_found:
        print(f"Đã xóa dữ liệu cho image_id {image_id}.")
    else:
        print(f"Không tìm thấy image_id {image_id} trong dữ liệu.")


def get_image_data(image_id, json_folder):
    json_files = ["train.json", "val.json", "test.json", "rel.json"]
    image_data = {}

    for file in json_files:
        file_path = os.path.join(json_folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if file in ["train.json", "val.json", "test.json"]:
            images = [img for img in data.get(
                "images", []) if img["id"] == image_id]
            annotations = [anno for anno in data.get(
                "annotations", []) if anno["image_id"] == image_id]
            if images:
                image_data["images"] = images
                image_data["annotations"] = annotations

        elif file == "rel.json":
            for key in ["train", "val", "test"]:
                if key in data and str(image_id) in data[key]:
                    image_data["relationships"] = data[key][str(image_id)]

    return image_data


def add_image_data(image_data, json_folder):
    with open(image_data, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    with open(json_folder + 'train.json', "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(json_folder + 'rel.json', "r", encoding="utf-8") as f:
        rel = json.load(f)

    for i in new_data:
        if "images" in i:
            data["images"].extend(i["images"])
        if "annotations" in i:
            data["annotations"].extend(i["annotations"])
        new_rel = {
            str(i["images"][0]['id']): i["relationships"]
        }
        rel['train'].update(new_rel)

    with open(json_folder + 'train.json', "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    with open(json_folder + 'rel.json', "w", encoding="utf-8") as f:
        json.dump(rel, f, indent=4)
    print("Dữ liệu ảnh đã được thêm thành công.")


# Ví dụ sử dụng
json_folder = "data/vg_focused/"
images_folder = "data/vg_focused/images"

image_id = [
    
]
print(len(image_id))
for i in image_id:
    delete_image_data(i, json_folder, images_folder)