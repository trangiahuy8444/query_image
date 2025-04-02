import os
import shutil

image_id = [
        1591956,
2409000	,
2376708,
1159544,
1159987,
2411203,
497950,
713202,
1592132,
2414568,
285612,
2408591,
2398480,
2413901,
150436,
498106,
2343671,
2408018,
2407207,
2404181,
2400840,
61599,
497956,
2409000,
2376708,
1591956,
713202,
1159544,
2414568,
2408591,
2366066,
2363398,
150309,
498106,
2369117,
2408778,
2377872,
713596,
713622,
1159987,
1592132,
2411203,
2410500,
2400722,
2398659,
2398480,
2384063,
2381785,
1592018,
2414928,
2413901,
2411391,
2410217,
4484,
107899,
150349,
150436,
285612,
497950,
2343671,
2408018,
2407207,
2406219,
2404181,
2400840,
2355212,
2343847
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