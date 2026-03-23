'''该文件未成功运行，原因为pycolmap/colmap不能直接处理h5格式文件'''


import pycolmap
import os

# 路径设置
image_dir = r"D:\COLMAP\project1\imagex_try"
output_dir = r"D:\COLMAP\project1\results_superpoint+lightglue_matching_lowres_quality_high"
database_path = os.path.join(output_dir, "database.db")

print("=" * 50)
print("开始重建")
print("=" * 50)

# ==================== 1. 删除旧的数据库 ====================
if os.path.exists(database_path):
    os.remove(database_path)
    print("删除旧数据库")

# ==================== 2. 创建新数据库 ====================
print("\n[1/3] 创建数据库...")
pycolmap.Database.create(database_path)

# ==================== 3. 导入特征和匹配 ====================
print("\n[2/3] 导入特征和匹配...")

# 打开数据库
db = pycolmap.Database(database_path)

# 导入特征
db.import_features(os.path.join(output_dir, "features.h5"))

# 导入匹配
db.import_matches(os.path.join(output_dir, "matches.h5"))

db.close()

print("✅ 导入完成")

# ==================== 4. 运行重建 ====================
print("\n[3/3] 运行增量重建...")

# 增量重建
reconstruction = pycolmap.incremental_mapping(
    database_path=database_path,
    image_path=image_dir,
    output_path=output_dir,
    options={  # 可选参数
        "map_builder": {
            "num_threads": 4,
        },
        "triangulation": {
            "min_tri_angle_deg": 5.0,
        },
    },
)

if reconstruction:
    print(f"\n✅ 重建成功！")
    print(f"   - 3D 点数量: {reconstruction.num_points3D()}")
    print(f"   - 相机数量: {reconstruction.num_reg_images()}")
    
    # 保存为文本格式
    reconstruction.write_text(os.path.join(output_dir, "sparse", "0"))
    print(f"\n📁 结果保存在: {output_dir}\\sparse\\0\\")
    
    # 查看 points3D.txt
    points3d_path = os.path.join(output_dir, "sparse", "0", "points3D.txt")
    if os.path.exists(points3d_path):
        with open(points3d_path, 'r') as f:
            lines = [l for l in f if not l.startswith('#')]
        print(f"\n🎉 共有 {len(lines)} 个3D点")
else:
    print("\n❌ 重建失败")