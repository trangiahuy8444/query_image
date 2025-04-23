import cv2
import numpy as np
import os

# Hàm tính khoảng cách Euclid giữa hai ảnh
def euclidean_distance(image1, image2):
    # Chuyển ảnh về grayscale nếu chưa
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Reshape ảnh thành vector một chiều
    image1 = image1.flatten()
    image2 = image2.flatten()

    # Tính khoảng cách Euclid
    distance = np.linalg.norm(image1 - image2)
    return distance

# Đọc ảnh input
image_input = cv2.imread('image_test/2403739.jpg')

# Thư mục chứa các ảnh output
output_folder = 'output_images/2403739'

# Lấy danh sách tất cả các file trong thư mục output_images
output_files = os.listdir(output_folder)

# Danh sách các ảnh output đã được đọc vào từ thư mục
output_images = []

# Đọc tất cả các ảnh trong thư mục và thay đổi kích thước cho khớp với ảnh input
for file_name in output_files:
    file_path = os.path.join(output_folder, file_name)
    # Kiểm tra xem file có phải là ảnh không (định dạng .jpg, .png, v.v...)
    if file_name.endswith(('.jpg', '.png', '.jpeg')):
        image = cv2.imread(file_path)

        # Thay đổi kích thước ảnh output sao cho khớp với kích thước ảnh input
        image_resized = cv2.resize(image, (image_input.shape[1], image_input.shape[0]))

        output_images.append((file_name, image_resized))

# Tính khoảng cách Euclid giữa ảnh input và từng ảnh output
distances = []
for file_name, img in output_images:
    distance = euclidean_distance(image_input, img)
    distances.append((file_name, distance))

# Sắp xếp các khoảng cách theo thứ tự tăng dần
distances.sort(key=lambda x: x[1])

# Lấy top 10, top 20, top 30 ảnh có khoảng cách Euclid nhỏ nhất
top_10 = distances[:10]
top_20 = distances[:20]
top_30 = distances[:30]

# In kết quả Top 10, Top 20 và Top 30
print("Top 10 ảnh tương tự nhất:")
for image_name, distance in top_10:
    print(f'{image_name}: {distance}')

print("\nTop 20 ảnh tương tự nhất:")
for image_name, distance in top_20:
    print(f'{image_name}: {distance}')

print("\nTop 30 ảnh tương tự nhất:")
for image_name, distance in top_30:
    print(f'{image_name}: {distance}')