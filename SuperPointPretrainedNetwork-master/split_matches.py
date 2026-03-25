'''匹配文件拆分脚本'''

import h5py
import os
import numpy as np
import glob

# ==================== 路径设置 ====================
matches_file = r"D:\COLMAP\project1\results_superpoint+lightglue_matching_lowres_quality_high\matches.h5"
output_dir = r"D:\COLMAP\project1\results_superpoint+lightglue_matching_lowres_quality_high"
matches_import_dir = os.path.join(output_dir, "matches_import")

# ==================== 步骤1：创建匹配文件 ====================
#print("=" * 50)
print("步骤1：生成匹配文件")
#print("=" * 50)

os.makedirs(matches_import_dir, exist_ok=True)
print(f"✅ 创建文件夹: {matches_import_dir}")

print("正在读取 matches.h5...")

with h5py.File(matches_file, 'r') as f:
    image_names = list(f.keys())
    print(f"找到 {len(image_names)} 张图像")
    
    total_matches = 0
    match_pairs = []  # 记录所有匹配对
    
    for img_name in image_names:
        matches_dict = f[img_name]
        
        for other_img_name in matches_dict.keys():
            matches = matches_dict[other_img_name][:]  # (N, 2)
            
            if len(matches) == 0:
                continue
            
            # 生成文件名
            img1_base = img_name.replace('.jpg', '')
            img2_base = other_img_name.replace('.jpg', '')
            match_file = os.path.join(matches_import_dir, f"{img1_base}-{img2_base}.txt")
            
            # 保存匹配点对
            np.savetxt(match_file, matches, fmt='%d', delimiter=' ')
            
            total_matches += len(matches)
            match_pairs.append((img_name, other_img_name))
            print(f"  {img_name} ↔ {other_img_name}: {len(matches)} 对匹配")
    
    print(f"\n✅ 生成 {len(match_pairs)} 个匹配文件，共 {total_matches} 对匹配")

# ==================== 步骤2：生成 match_list.txt ====================
#print("\n" + "=" * 50)
print("步骤2：生成 match_list.txt")
#print("=" * 50)

match_files = glob.glob(os.path.join(matches_import_dir, "*.txt"))
match_list_path = os.path.join(output_dir, "match_list.txt")

with open(match_list_path, 'w') as f:
    for match_file in match_files:
        base = os.path.basename(match_file).replace('.txt', '')
        img1, img2 = base.split('-')
        f.write(f"{img1}.jpg {img2}.jpg\n")

print(f"✅ 生成 {match_list_path}")
print(f"   共 {len(match_files)} 对匹配")

#print("\n" + "=" * 50)
print("🎉 全部完成！")
print(f"匹配文件: {matches_import_dir}")
print(f"匹配列表: {match_list_path}")
#print("=" * 50)

