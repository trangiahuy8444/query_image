import os
from pathlib import Path

def compare_folders(folder1_path, folder2_path):
    # Chuyển đổi đường dẫn thành Path objects
    folder1 = Path(folder1_path)
    folder2 = Path(folder2_path)
    
    # Kiểm tra xem các folder có tồn tại không
    if not folder1.exists():
        print(f"Folder {folder1_path} không tồn tại!")
        return
    if not folder2.exists():
        print(f"Folder {folder2_path} không tồn tại!")
        return
    
    # Lấy danh sách tên file từ mỗi folder và bỏ định dạng .jpg
    files1 = {f.stem for f in folder1.glob("*.jpg")}
    files2 = {f.stem for f in folder2.glob("*.jpg")}
    
    # Tìm các file có trong folder1 mà không có trong folder2
    files_only_in_folder1 = files1 - files2
    print(len(files_only_in_folder1))
    # In kết quả
    for file in sorted(files_only_in_folder1):
        print(f'{file},')

if __name__ == "__main__":
    # Ví dụ sử dụng
    folder1 = "./images"
    folder2 = "./output_images/286068"
    compare_folders(folder1, folder2) 