''' 生成3D地图，为 3D 点生成描述子 '''
''' 于2026/3/26晚上跑通 '''

#文件读取 头文件
import h5py
import numpy as np
import pycolmap
#from demo_superpoint import SuperPointFrontend

#import cv2
#import os


if __name__ == "__main__":


    print("=" * 50)
    print("第1步：为 3D 点生成描述子")
    print("=" * 50)

    # 路径
    recon_dir = r"D:\COLMAP\project1\hloc_output\reconstruction" # output_dir / "reconstruction" 
    features_file = r"D:\COLMAP\project1\hloc_output\feats-superpoint-n4096-r1024.h5" #features_path = output_dir / "feats-superpoint-n4096-r1024.h5"

    # 加载 3D 地图
    print("加载 3D 地图...")
    reconstruction = pycolmap.Reconstruction() #建一个空的地图容器
    reconstruction.read_binary(recon_dir)
    print(f"✅ 加载成功")
    print(f"3D 点数量: {reconstruction.num_points3D()}")
    print(f"图像数量: {len(reconstruction.images)}")

    # 读取 HLOC 提取的特征
    with h5py.File(features_file, 'r') as f:
        feature_names = set(f.keys()) #f.keys() 是一个可迭代对象，包含了所有的图像名（字符串），set() 将其转换为集合，方便后续查找
        print(f"特征文件中有 {len(feature_names)} 张图片")

    # 为每个 3D 点找对应的描述子（important）
    point3d_to_desc = {}

    print("\n处理中...")
    processed = 0
    for img_id, img in reconstruction.images.items(): # img 是 pycolmap.Image 对象，里面有图像的所有信息，包括 2D 点和对应的 3D 点
        # 安全检查：防止reconstruction中有些图像不在features.h5文件中
        img_name = img.name
        if img_name not in feature_names:
            print(f"  跳过 {img_name}（不在特征文件中）")
            continue
        '''
        .h5 文件结构：
        feats-superpoint-n4096-r1024.h5
        N: 这张图片的特征点数量
        ├── 1.jpg
        │   ├── keypoints (N×2)      关键点坐标
        │   ├── descriptors (256×N)  描述子（每一个特征点有一个256维描述子）
        |   └── score (N×1)          特征点置信度为一个[0, 1]的值
        ├── 2.jpg
        │   ├── keypoints (N×2) 
        │   ├── descriptors (256×N)
        |   └── score (N×1)
        └── ...
        '''
        # 读取特征
        # 从 .h5 里把这张图对应的所有特征点坐标（keypoints）和所有描述子（descriptors）取出来, 并对描述子矩阵进行转置
        with h5py.File(features_file, 'r') as f:
            keypoints = f[img_name]['keypoints'][:]  # N×2
            descriptors = f[img_name]['descriptors'][:].T  # N×256

        print(f"  {img_name}: {len(keypoints)} 个特征点") #len(keypoints) 是这张图的特征点数量，len(descriptors) 也是一样的，因为每个特征点都有一个描述子

        # ======================== 下面是重点 =========================
        # 遍历该图像的所有 2D 特征点，如果这个 2D 点没有对应的 3D 点（比如重建失败的点），就跳过。
        for point2d in img.points2D:
            if not point2d.has_point3D(): # has_point3D() 是 pycolmap.Point2D 的方法，返回 True/False，表示这个 2D 点是否有对应的 3D 点
                continue
            point3d_id = point2d.point3D_id # point2d.point3D_id: point2D的坐标对应的point3D的ID
            # 但是这个赋值真的有必要吗？
            
            # 找到这个 2D 特征点在 SuperPoint 特征中的索引
            # COLMAP 记录的 2D 点坐标 (x, y) 是亚像素精度（浮点数），而 keypoints 里存的是整数像素（因为 SuperPoint 输出就是整数坐标）。
            x, y = point2d.xy # point2d.xy: point2D的坐标

            # 计算所有 SuperPoint 特征点到这个 2D 点的距离，找到最近的那个特征点
            distances = np.linalg.norm(keypoints - np.array([x, y]), axis=1) # np.linalg.norm：算这个浮点坐标到所有整数特征点的欧氏距离
            closest_idx = np.argmin(distances) # np.argmin：找到离得最近的那个整数特征点的索引（下标）。
            
            if distances[closest_idx] < 5:  # 5 像素内
                point3d_to_desc[point3d_id] = descriptors[closest_idx]

            '''
            当下的问题：
            1. 如果一个 3D 点在多张图像中都有对应的 2D 点，那么它会被多次赋值，最后只会保留最后一次的描述子。
            2. 如果想要为每个 3D 点生成一个更稳定的描述子，可以考虑对所有对应的 2D 点的描述子进行平均或者其他聚合操作。
            3. 这里的距离阈值（5 像素）可以根据实际情况调整，太大可能会匹配到错误的特征点，太小可能会漏掉一些正确的匹配。

            解决方案伪代码：取平均（同一个3D点对应的不同的2D特征点的描述子貌似是相似的？）
            if point3d_id in point3d_to_desc:
                point3d_to_desc[point3d_id] = (point3d_to_desc[point3d_id] + descriptors[closest_idx]) / 2
            else:
                point3d_to_desc[point3d_id] = descriptors[closest_idx]
            '''
        
        processed += 1
        if processed % 10 == 0:
            print(f"  已处理 {processed} 张图片")

    print(f"\n为 {len(point3d_to_desc)} 个 3D 点生成描述子")


    # 保存
    np.save('point3d_to_desc.npy', point3d_to_desc)
    print("✅ 保存到 point3d_to_desc.npy")
    # 问题：未指定输出路径
    # 这个 point3d_to_desc.npy 文件是一个字典，key 是 3D 点的 ID，value 是对应的描述子（256维向量）。在后续的 PnP 定位中，我们可以用这个字典来查找每个 3D 点的描述子，从而进行匹配和定位。
    # 未保存的点会不会影响后续的PnP定位？如果有些3D点没有描述子，那么在PnP阶段就无法使用这些点进行匹配，可能会导致定位失败或者精度下降.
