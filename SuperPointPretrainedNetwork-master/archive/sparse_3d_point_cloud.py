'''该文件未成功运行，原因为pycolmap/colmap不能直接处理h5格式文件'''

import pycolmap
import os

# 路径
output_dir = r"D:\COLMAP\project1\results_superpoint+lightglue_matching_lowres_quality_high"
image_dir = r"D:\COLMAP\project1\imagex_try"
database_path = os.path.join(output_dir, "database.db")

# 删除旧数据库
if os.path.exists(database_path):
    os.remove(database_path)
    print("✅ 已删除旧数据库")

# 创建新数据库
db = pycolmap.Database(database_path)
print("✅ 数据库创建成功")

# 添加所有图像
import glob
for img_path in glob.glob(os.path.join(image_dir, "*.jpg")):
    img_name = os.path.basename(img_path)
    db.add_image(img_name, camera_model="SIMPLE_RADIAL")
    print(f"  添加: {img_name}")

db.close()

# 导入特征（手动从 features.h5 导入到数据库）
import h5py
import numpy as np

# 读取 features.h5
features_file = os.path.join(output_dir, "features.h5")
with h5py.File(features_file, 'r') as f:
    # 获取所有图像名
    for img_name in f.keys():
        keypoints = f[img_name]['keypoints'][:]
        descriptors = f[img_name]['descriptors'][:]
        
        # 转换为 COLMAP 格式
        # keypoints 格式: (N, 2) -> x, y
        # descriptors 格式: (N, 256) -> 需要转置为 (256, N)
        
        print(f"  导入: {img_name}, {len(keypoints)} 个特征点")
        
        # 这里需要把特征点写入数据库
        # 但 pycolmap 4.0.2 可能没有直接写入的方法

print("特征导入完成（需要 pycolmap 支持）")

