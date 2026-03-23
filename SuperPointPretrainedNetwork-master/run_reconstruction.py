'''成功运行没报错'''

import deep_image_matching as dim
import os

# 设置路径
image_dir = r"D:\COLMAP\project1\imagex_try"
output_dir = r"D:\COLMAP\project1\output"

os.makedirs(output_dir, exist_ok=True)

# 创建配置字典（用 --images 或 --dir）
config_dict = {
    "images": image_dir,           # 用 "images" 而不是 "image_dir"
    "output_dir": output_dir,      # 输出目录
    "pipeline": "superpoint+lightglue",  # 特征+匹配方法
    "max_image_size": 1600,        # 可选：缩放图片
    "use_gpu": False,              # 可选：用 GPU
}

# 创建 Config 对象
config = dim.Config(config_dict)

# 创建 ImageMatcher
matcher = dim.ImageMatcher(config)

# 运行匹配
print("开始特征提取和匹配...")
matcher.run()

print("完成！结果保存在:", output_dir)

