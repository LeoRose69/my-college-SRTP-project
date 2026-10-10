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

    image_dir = Path(r"D:\COLMAP\project1\imagex_try") #图片文件夹
    output_dir = Path(r"D:\COLMAP\project1\hloc_output") #中间文件和最终地图放在哪里
    output_dir.mkdir(parents=True, exist_ok=True) #确保输出目录存在，如果不存在就创建它

    print("=" * 50) 
    print("HLOC 3D 重建")
    print("=" * 50)

    # ========== 特征提取 ==========
    print("\n[1/3] 提取 SuperPoint 特征...")
    feature_conf = extract_features.confs["superpoint_aachen"] #进行特征提取使用的配置菜单，适配superpoint的参数
    features_path = extract_features.main(
        conf=feature_conf,
        image_dir=image_dir,
        export_dir=output_dir,
        as_half=False,
    ) 
    #features_path 是一个 h5 文件，里面包含了所有图像的特征点和描述子；
    #返回路径：output_dir / "feats-superpoint-n4096-r1024.h5"
    print(f"特征文件: {features_path}") 

    # ========== 生成 pairs.txt ==========
    print("\n[2/3] 生成配对文件...")
    images = sorted([f.name for f in image_dir.glob("*.jpg")]) #images 是一个列表，包含了所有图片的文件名，按字母顺序排序
    pairs_file = output_dir / "pairs.txt"
    with open(pairs_file, 'w') as f: #打开 pairs.txt 文件准备写入
        for i, img1 in enumerate(images): #双层循环，生成所有图片的配对组合
            for img2 in images[i+1:]:
                f.write(f"{img1} {img2}\n")
    print(f"生成 {len(images)*(len(images)-1)//2} 对匹配")
    #告诉hloc的特征匹配器需要检查的组合

    # ========== 特征匹配 ==========
    print("\n[3/3] 用 LightGlue 匹配...")
    matcher_conf = match_features.confs["superpoint+lightglue"]
    matches_path = match_features.main( #使用pair.txt和features.h5文件进行特征匹配，生成matches.h5文件至output_dir
        conf=matcher_conf,
        pairs=pairs_file,
        export_dir=output_dir,
        features=features_path,        # 直接传 .h5 文件路径！
        matches=output_dir / "matches.h5",
    )

    # ========== 重建 ==========
    print("\n[4/3] 3D 重建...")
    model = reconstruction.main( #调用 pycolmap，使用特征和匹配文件进行 SfM 重建，生成点云和相机位姿
        sfm_dir=output_dir / "reconstruction", #输出结果存放路径
        image_dir=image_dir,
        pairs=pairs_file,
        #使用特征和匹配的结果进行重建
        features=features_path,        # 这里也要用文件路径
        matches=matches_path,
        #让程序自动猜测相机内参
        camera_mode="AUTO",
        #把重建过程的日志打印出来
        verbose=True,
    )

    #检查是否成功 - 基本是废的啦
    points3d_path = output_dir / "reconstruction" / "points3D.bin"
    if points3d_path.exists():
        with open(points3d_path, 'r') as f:
            lines = [l for l in f if not l.startswith('#')]
        print(f"\n🎉 成功！共有 {len(lines)} 个3D点")
    else:
        print("\n⚠️  points3D.txt 还没生成")


'''
在 output_dir / "reconstruction" 目录下：
(重要：)
cameras.bin - 重建的相机内参
images.bin - 重建的相机位姿
points3D.bin - 重建的3D点云
(不重要：)
rigs.bin - COLMAP自动生成的刚体组信息(用的是普通单相机，所以是空的，但 COLMAP 默认总会吐这个文件)
frames.bin - COLMAP自动生成的视频帧索引信息(可以忽略)
colmap.LOG.xxx.34016 - COLMAP的后台运行日志

'''

        

