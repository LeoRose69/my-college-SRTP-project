'''尝试解决问题：特征提取结果的EXIF信息中没有正确的旋转方向'''
import os
from PIL import Image

image_dir = r"D:\COLMAP\project1\imagex_try"

for img_file in os.listdir(image_dir):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_dir, img_file)
        
        # 打开图片
        img = Image.open(img_path)
        
        # 创建没有 EXIF 的新图片
        data = list(img.getdata())
        img_no_exif = Image.new(img.mode, img.size)
        img_no_exif.putdata(data)
        
        # 保存（覆盖原文件）
        img_no_exif.save(img_path)
        print(f"✅ 已清除 EXIF: {img_file}")
        