# RelTR Neo4j Application

Ứng dụng Flask kết hợp với Neo4j và mô hình RelTR để phân tích mối quan hệ trong hình ảnh.

## Triển khai lên Render

### Bước 1: Tạo tài khoản Render

1. Truy cập [Render](https://render.com/) và đăng ký tài khoản mới.
2. Kết nối tài khoản GitHub của bạn với Render.

### Bước 2: Tạo dịch vụ Web mới

1. Trong dashboard Render, nhấp vào "New" và chọn "Web Service".
2. Kết nối với repository GitHub của bạn.
3. Cấu hình dịch vụ:
   - **Name**: reltr-app (hoặc tên bạn muốn)
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free

### Bước 3: Cấu hình biến môi trường

Thêm các biến môi trường sau:
- `NEO4J_URI`: URI của cơ sở dữ liệu Neo4j của bạn
- `NEO4J_USERNAME`: Tên người dùng Neo4j
- `NEO4J_PASSWORD`: Mật khẩu Neo4j

### Bước 4: Triển khai

1. Nhấp vào "Create Web Service".
2. Render sẽ tự động triển khai ứng dụng của bạn.

### Bước 5: Cấu hình GitHub Actions (tùy chọn)

1. Trong repository GitHub của bạn, đi đến "Settings" > "Secrets and variables" > "Actions".
2. Thêm các secret sau:
   - `RENDER_API_KEY`: API key từ Render (lấy từ Account Settings > API Keys)
   - `RENDER_SERVICE_ID`: ID của dịch vụ Render (lấy từ URL của dịch vụ)

## Triển khai thủ công

Nếu bạn muốn triển khai thủ công, hãy làm theo các bước sau:

1. Cài đặt các phụ thuộc:
   ```
   pip install -r requirements.txt
   ```

2. Chạy ứng dụng:
   ```
   gunicorn app:app
   ```

## Cấu trúc thư mục

- `app.py`: File chính của ứng dụng Flask
- `requirements.txt`: Danh sách các phụ thuộc Python
- `RelTR/`: Thư mục chứa mô hình RelTR
- `data/`: Thư mục chứa dữ liệu hình ảnh
- `uploads/`: Thư mục để lưu ảnh upload
- `output_images/`: Thư mục để lưu ảnh output 