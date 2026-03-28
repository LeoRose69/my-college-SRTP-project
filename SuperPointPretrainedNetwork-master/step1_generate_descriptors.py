''' 生成3D地图，为 3D 点生成描述子 '''
''' 于2026/3/26晚上跑通 '''

import h5py
import numpy as np
import pycolmap
from demo_superpoint import SuperPointFrontend
#import cv2
#import os


if __name__ == "__main__":


    print("=" * 50)
    print("第1步：为 3D 点生成描述子")
    print("=" * 50)

    # 路径
    recon_dir = r"D:\COLMAP\project1\hloc_output\reconstruction"
    features_file = r"D:\COLMAP\project1\hloc_output\feats-superpoint-n4096-r1024.h5"

    # 加载 3D 地图
    print("加载 3D 地图...")
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read_binary(recon_dir)
    print(f"✅ 加载成功")
    print(f"3D 点数量: {reconstruction.num_points3D()}")
    print(f"图像数量: {len(reconstruction.images)}")

    # 读取 HLOC 提取的特征
    with h5py.File(features_file, 'r') as f:
        feature_names = set(f.keys())
        print(f"特征文件中有 {len(feature_names)} 张图片")

    # 为每个 3D 点找对应的描述子
    point3d_to_desc = {}

    print("\n处理中...")
    processed = 0
    for img_id, img in reconstruction.images.items():
        img_name = img.name
        if img_name not in feature_names:
            print(f"  跳过 {img_name}（不在特征文件中）")
            continue
        
        # 读取特征
        with h5py.File(features_file, 'r') as f:
            keypoints = f[img_name]['keypoints'][:]  # N×2
            descriptors = f[img_name]['descriptors'][:].T  # N×256

        print(f"  {img_name}: {len(keypoints)} 个特征点")
        
        # 遍历该图像的所有 2D 特征点
        for point2d in img.points2D:
            if not point2d.has_point3D():
                continue
            
            point3d_id = point2d.point3D_id
            
            # 找到这个 2D 点在 SuperPoint 特征中的索引
            x, y = point2d.xy
            distances = np.linalg.norm(keypoints - np.array([x, y]), axis=1)
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] < 5:  # 5 像素内
                point3d_to_desc[point3d_id] = descriptors[closest_idx]
        
        processed += 1
        if processed % 10 == 0:
            print(f"  已处理 {processed} 张图片")

    print(f"\n为 {len(point3d_to_desc)} 个 3D 点生成描述子")


    # 保存
    np.save('point3d_to_desc.npy', point3d_to_desc)
    print("✅ 保存到 point3d_to_desc.npy")

