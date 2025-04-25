FROM python:3.9-slim

WORKDIR /app

# Cài đặt các phụ thuộc hệ thống
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Sao chép các file cần thiết
COPY requirements.txt .
COPY app.py .
COPY RelTR/ RelTR/
COPY templates/ templates/
COPY static/ static/

# Cài đặt các phụ thuộc Python
RUN pip install --no-cache-dir -r requirements.txt

# Tạo các thư mục cần thiết
RUN mkdir -p uploads output_images

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Mở cổng
EXPOSE 10000

# Chạy ứng dụng
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"] 