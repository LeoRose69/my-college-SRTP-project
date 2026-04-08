# 以下内容是对项目内杂乱的文件的解释，项目的演变过程，以方便作者理解

## 初始版：superpoint_pretrained_network
- demo_superpoint.py
- superpoint_v1.pth

superpoint pretrained network下载自github，可以进行图片**特征提取**，且可以处理相机传入的视频流。
需要在工作区内包含superpoint_v1.pth才可正常工作

## 废案
> 注意： 废案中所提到的所有文件都已经没有用了。废案的输出结果results_superpoint+lightglue_matching_lowres_quality_high文件夹和里面的东西也都没用了。
### 一、棋盘定位废案
- online_pnp.py
  
#### 总结
*deepseek是纯废物*

### 二、直接使用COLMAP跑完整个过程的废案
COLMAP在特征提取和匹配的时候使用的是SIFT算法暴力解算，在运行的时候GUI总是**闪退**

### 三、deep-image-matching + COLMAP/pycolmap 废案
*主要的：*
- run_reconstruction.py
- sparse_3D_point_cloud.py

*为了疯狂修bug而创建的：*
- split_features.py
- split_matches.py
- clear_EXIF.py
- test.py

#### 思路
1. deep-image-matching包：特征提取与特征匹配（run_reconstruction.py）
   - 内嵌有superpoint lightglue
   - 可以成功运行，解算出的结果存放在：results_superpoint+lightglue_matching_lowres_quality_high中。
  
2. pycolmap包：3D点云重建与相机位姿解算（sparse_3D_point_cloud.py）
   - 集成了COLMAP的核心代码
   - 运行失败

#### 问题
1. DIM解算出来的结果（特征提取、匹配的结果）是完整的h5类型文件: features.h5和matches.h5；但是COLMAP进行3D稀疏点云地图重建的核心算法不能接受这个格式的结果。
2. 更难解决的问题：COLMAP的数据库不接受256维的描述子

#### 尝试解决
1. 使用两个split.py文件来分割h5文件，企图得到能够用pycolmap来处理的特征点格式。
2. 后来放弃了pycolmap，尝试直接使用COLMAP（在代码中调用COLMAP.bat）。但是COLMAP功能集成度太高了，最终失败了（具体体现为缺失EXTI），依然没能成功重建3D稀疏点云。
3. clear_EXIF.py：尝试解决EXIF问题，最后的挣扎。
4. COLMAP不能处理256维的描述子，这个问题彻底杀死了这个西拼东凑的方案。

#### 总结
1. 本人对DIM和pycolmap并不熟悉。使用太多松散的模块，在功能衔接上未免会出现问题。
2. COLMAP的功能的集成度实在是太高了。从这个事件我们可以学习到，若想绕开SIFT的同时去用sfm来做三维重建，想用COLMAP是不太可能的。
3. **核心与关键：** 既然DIM + COLMAP是失败的，那为什么不用DIM + Pycolmap？原因在于中间的格式转换太费劲。DIM主要是为了COLMAP而诞生的，而不是pycolmap；所以说DIM的输出格式对于COLMAP.bat更友好。
   
   但由于一些数据库的致命问题，显然DIM + COLMAP还是不可行的。故被放弃。

   综上，DIM与HLOC相比：还是太割裂了。HLOC与pycolmap的配合更好。

## 最终方案（第一代）：改用HLOC
- run_hloc.py
- step1_generate_descriptors.py
- step2_pnp_localization.py

### 一、特征提取和匹配：run_hloc.py

1. HLOC的好处在于，它提供可以调用pycolmap的API，相当于是DIM的替代工具。除此之外没什么可说的。依旧是：
   - superpoint特征提取 
   - lightglue特征匹配 
   - pycolmap(sfm)3D稀疏点云重建
  
2. **跑出的结果：** 特征提取和匹配（前端）跑出的结果（算是中间文件产物）都存放在hloc_output里。**与图片文件夹在同一根目录下**。这次的运行结果很顺利，无报错。其中有输出的
   - 特征点和匹配结果的h5文件
   - 相机参数和位姿解算结果（bin文件）
   - 计算出的3D点云（3Dpoints.bin）
   - 中间产物database文件
3. 3D地图已建成，如果有COLMAP GUI的话，理论上这个时候可以进行稀疏点云的可视化了。
   - 点击 File → Import model
   - 选择 reconstruction 文件夹（里面有 cameras.bin、images.bin、points3D.bin）
   - 若弹窗说没有ini文件或说没有valid database and image path的话：选上就可以看selected image的位姿；不选就只看3D点云图。

### 二、为3D点云生成描述子：step1_generate_descriptors.py

1. 这个过程是为了用户端进行pnp匹配之前所进行的最后一步，只是一小步但是很重要。
2. 输出结果：point3d_to_desc.npy 文件，在代码文件夹的根目录下，不可查看。

### 三、pnp解算：step2_pnp_localization.py

1. 仿照了superpoint的demo代码的结构写的demo。写了三种demo。
2. 输出：
   - 单张图片：代码文件夹里
   - 多张图片：输入的图片文件夹**内**
   - 未完待续

### 未来想法与当前问题

1. 未来可能想办法尝试将几个文件合并
2. 该版本(small update)的情况：
   - 调整了输出图片的坐标系的坐标轴的长度，让它更明显
   - **camera确认可以运行，不会报错**
   - 代码效率太低
#### 目前版本代码效率问题分析：
1. 当前代码的主要瓶颈：
   - 特征提取（最耗时）：SuperPointFrontend 的神经网络推理
   - 特征匹配：代码第61-65行的暴力匹配（O(n×m) 复杂度），如果有 500个图像特征 和 10000个地图3D点，就要做 500万次点积。
   - PnP 求解：RANSAC 1000次迭代

2. 改善方案：
   - 用 KD-Tree/Annoy 替代暴力匹配 ✅ 最有效；这样能快 100倍。
   - 降低 SuperPoint 输入分辨率 ✅ 次有效；改 max_size 从 1024 降到 512 或更低。会减少特征点数量，加快整个流程，但可能降低精度。
   - 减少 PnP RANSAC 迭代次数

   ```python
   iterationsCount=500,  # 改小一半
   reprojectionError=15.0,  # 允许更大误差
   ```
   - 目标：10+ FPS
