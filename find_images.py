import os
import shutil

image_id = [
]
# Đảm bảo thư mục output tồn tại
os.makedirs('images', exist_ok=True)
# Duyệt qua tất cả ảnh trong images_folder
for file_name in os.listdir('vg_1000/images'):
    image, ext = os.path.splitext(file_name)  # Tách id ảnh và đuôi file
    # Nếu id có trong danh sách, sao chép ảnh sang output_folder
    image_id = list(map(str, image_id))
    if image in image_id:
        src_path = os.path.join('vg_1000/images', file_name)
        dst_path = os.path.join('images', file_name)
        shutil.copy2(src_path, dst_path)  # copy2 để giữ nguyên metadata
        print(f"Đã sao chép: {file_name}")
print("Hoàn thành việc trích xuất ảnh!")