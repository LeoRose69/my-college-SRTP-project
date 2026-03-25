'''
重新尝试使用hloc:直接用hloc的接口，不再自己拆分特征和匹配文件，减少出错点
改动1：用lightglue替换superglue,因为superglue的python接口不太好用，且lightglue性能更好

注意事项：
各种类可以接受的参数类型
'''

'''
    print("\n=== 调试信息 ===")
    print(f"features_path 类型: {type(features_path)}")
    print(f"features_path 值: {features_path}")
    print(f"是否存在: {features_path.exists()}")
    print(f"是文件吗: {features_path.is_file()}")
    print(f"是文件夹吗: {features_path.is_dir()}")
    print(f"拼接 pairs.txt: {features_path / 'pairs.txt'}")
    print(f"该文件存在吗: {(features_path / 'pairs.txt').exists()}")
    print("=== 调试结束 ===\n")

    '''

    #运行出来的features_path是一个h5文件，里面包含了所有图像的特征点和描述子
    #features_path 类型: <class 'pathlib._local.WindowsPath'>
    #features_path 值: D:\COLMAP\project1\hloc_output\feats-superpoint-n4096-r1024.h5        

    
    # ========== 手动生成 pairs.txt ==========
'''
    1. 生成配对文件 pairs.txt，服务于特征匹配/点云重建
    2. pairs.txt 格式：每行两个图片文件名，表示一对
    
    #获取文件夹里所有 .jpg 图片的名字，排序后存在列表里。例如 ['1.jpg', '2.jpg', '3.jpg']
    #指定输出文件路径，比如 D:\...\hloc_output\pairs.txt
    #打开文件准备写入（'w' 表示写入，如果文件存在会覆盖）
        #循环遍历每张图片，i 是索引，img1 是图片名  
            #内层循环，从 img1 后面的图片开始遍历，避免重复配对
                #写入一行，格式是 图片A 图片B，例如 1.jpg 2.jpg
    #检索：打印总共生成了多少对（组合数公式 n*(n-1)/2）
    '''

import os
from pathlib import Path
from hloc import extract_features, match_features, reconstruction

   
# ==================== 入口保护 ====================
if __name__ == "__main__":
    image_dir = Path(r"D:\COLMAP\project1\imagex_try")
    output_dir = Path(r"D:\COLMAP\project1\hloc_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("HLOC 3D 重建")
    print("=" * 50)

    # ========== 特征提取 ==========
    print("\n[1/3] 提取 SuperPoint 特征...")
    feature_conf = extract_features.confs["superpoint_aachen"]
    features_path = extract_features.main(
        conf=feature_conf,
        image_dir=image_dir,
        export_dir=output_dir,
        as_half=False,
    )
    print(f"特征文件: {features_path}")

    # ========== 生成 pairs.txt ==========
    print("\n[2/3] 生成配对文件...")
    images = sorted([f.name for f in image_dir.glob("*.jpg")])
    pairs_file = output_dir / "pairs.txt"
    with open(pairs_file, 'w') as f:
        for i, img1 in enumerate(images):
            for img2 in images[i+1:]:
                f.write(f"{img1} {img2}\n")
    print(f"生成 {len(images)*(len(images)-1)//2} 对匹配")

    # ========== 特征匹配 ==========
    print("\n[3/3] 用 LightGlue 匹配...")
    matcher_conf = match_features.confs["superpoint+lightglue"]
    matches_path = match_features.main(
        conf=matcher_conf,
        pairs=pairs_file,
        export_dir=output_dir,
        features=features_path,        # 直接传 .h5 文件路径！
        matches=output_dir / "matches.h5",
    )

    # ========== 重建 ==========
    print("\n[4/3] 3D 重建...")
    model = reconstruction.main(
        sfm_dir=output_dir / "reconstruction",
        image_dir=image_dir,
        pairs=pairs_file,
        features=features_path,        # 这里也要用文件路径
        matches=matches_path,
        camera_mode="AUTO",
        verbose=True,
    )

    points3d_path = output_dir / "reconstruction" / "0" / "points3D.txt"
    if points3d_path.exists():
        with open(points3d_path, 'r') as f:
            lines = [l for l in f if not l.startswith('#')]
        print(f"\n🎉 成功！共有 {len(lines)} 个3D点")
    else:
        print("\n⚠️  points3D.txt 还没生成")

        

