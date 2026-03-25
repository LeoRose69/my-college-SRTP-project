'''
尝试删除旧数据库，创建新数据库，并添加图像信息
结果删除完之后代码开始报错了
COLMAP只接受txt格式的特征，且无法处理256维的描述子
下一步：
1. 尝试COLMAP SIFT跑全套
2. 尝试改成hloc
'''

import pycolmap
import os

# 路径
output_dir = r"D:\COLMAP\project1\results_superpoint+lightglue_matching_lowres_quality_high"
image_dir = r"D:\COLMAP\project1\imagex_try"
database_path = os.path.join(output_dir, "database.db")

# 删除旧数据库
if os.path.exists(database_path):
    os.remove(database_path)
    print("删除旧数据库")

# 创建新数据库
db = pycolmap.Database(database_path)
print("✅ 数据库创建成功")

# 添加所有图像
import glob
for img_path in glob.glob(os.path.join(image_dir, "*.jpg")):
    img_name = os.path.basename(img_path)
    # 添加图像，指定相机模型
    db.add_image(img_name, camera_model="SIMPLE_RADIAL")
    print(f"  添加: {img_name}")

db.close()
print("✅ 图像信息添加完成")
