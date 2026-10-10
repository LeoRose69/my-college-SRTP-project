'''转格式'''
import h5py
import os
import numpy as np

# 路径
features_file = r"D:\COLMAP\project1\results_superpoint+lightglue_matching_lowres_quality_high\features.h5"
output_dir = r"D:\COLMAP\project1\results_superpoint+lightglue_matching_lowres_quality_high\features_import"
image_dir = r"D:\COLMAP\project1\imagex_try"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

print("正在读取 features.h5...")

# 读取并拆分
with h5py.File(features_file, 'r') as f:
    # 获取所有图像名
    image_names = list(f.keys())
    print(f"找到 {len(image_names)} 张图像")
    
    for img_name in image_names:
        # 读取特征点 (N, 2) 和描述子 (N, 256)
        keypoints = f[img_name]['keypoints'][:]
        descriptors = f[img_name]['descriptors'][:]
        
        # 保存为单独的文件
        out_file = os.path.join(output_dir, f"{img_name}.h5")
        with h5py.File(out_file, 'w') as out:
            out.create_dataset('keypoints', data=keypoints)
            out.create_dataset('descriptors', data=descriptors)
        
        print(f"✅ {img_name}: {len(keypoints)} 个特征点")

print(f"\n完成！文件保存在: {output_dir}")

