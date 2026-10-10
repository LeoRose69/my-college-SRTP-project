'''
类名	            职责（大白话）	                                                对应实物比喻

MapManager	        加载离线地图（3D坐标+指纹），并建立快速检索索引	                  离线地图SD卡
FeatureExtractor	处理当前相机拍到的图片，提取 2D 特征点和指纹	                  摄像头传感器
PnPSolver	        根据 2D-3D 点对，解算相机位姿（位置+朝向），并画坐标轴	          定位解算芯片
Visualizer	        在图像上写字、画点	                                            屏幕显示器
Localizer	        总指挥官，把上面所有零件组装起来，处理单张图、文件夹、摄像头	    飞控主板
'''



import pycolmap
import numpy as np
import cv2
from demo_superpoint import SuperPointFrontend

import argparse
import os
#from datetime import datetime
import glob

import gc   #内存管理
from scipy.spatial.transform import Rotation as R
from annoy import AnnoyIndex



# ==================== map_manager.py ====================

class MapManager:
    """管理 3D 地图

    传入形参：
    map_path: 3D地图路径(reconstruction文件夹)
    desc_path: 描述子路径(point3d_to_desc.npy)

    成员变量：
    reconstruction: pycolmap.Reconstruction 对象，包含 3D 点和相机参数
    point3d_coords: dict，key=3D点ID，value=3D点坐标(x,y,z)
    point3d_to_desc: dict，key=3D点ID，value=描述子（256维向量）
    annoy_index: Annoy 索引，用于快速检索最近邻的 3D 点
    pid_to_idx: dict，key=Annoy索引，value=3D点ID，用于从 Annoy 索引找到对应的 3D 点
    K: 相机内参矩阵

    接口：
    get_camera_intrinsics(): 返回相机内参矩阵 K
    match_3d_points(pts, desc): 给定 2D 特征点和描述子，返回匹配的 2D-3D 点对

    """

    # map_path: 3D地图路径(reconstruction文件夹)
    # desc_path: 描述子路径(point3d_to_desc.npy)
    def __init__(self, map_path, desc_path):     # 构造函数
        self.reconstruction = pycolmap.Reconstruction()
        self.reconstruction.read_binary(map_path)

        # 3D 点坐标
        self.point3d_coords = {} # 先创建一个空字典
        # key是3D点的ID，value是3D点的坐标(x,y,z)
        for pid, p3d in self.reconstruction.points3D.items():
            self.point3d_coords[pid] = p3d.xyz

        # 描述子映射 + Annoy 索引
        self.point3d_to_desc = np.load(desc_path, allow_pickle=True).item() 
        # 读取 point3d_to_desc.npy 文件，
        # 还原得到一个字典，key是3D点的ID，value是对应的描述子（256维向量）

        # 建立 Annoy 索引（加速特征匹配）
        print("建立 Annoy 索引...")
        self.annoy_index = AnnoyIndex(256, metric='angular')  # 256维，
        # metric='angular' 表示用角度距离（点积）来衡量相似度，适合描述子匹配

        # 映射：Annoy索引 -> 3D点ID
        self.pid_to_idx = {}  
        for idx, (pid, pdesc) in enumerate(self.point3d_to_desc.items()): #遍历描述子字典
            self.annoy_index.add_item(idx, pdesc) #把每个描述子加入 Annoy 索引，idx 是 Annoy 的索引，pdesc 是描述子
            self.pid_to_idx[idx] = pid #把 Annoy 索引映射到 3D 点 ID，方便后续查找
        self.annoy_index.build(30)  # 30棵树，平衡精度和速度
        print(f"✅ Annoy 索引建立完成（30棵树）")

        # 相机内参
        cam = next(iter(self.reconstruction.cameras.values())) # 从地图里随便抓一张相机的参数出来（假设所有图片用的是同一台相机，焦距相同）。
        # 注意：假设COLMAP重建的相机是针孔模型(SIMPLE_PINHOLE)，且fx=fy，cx,cy是主点坐标
        # HLOC通常默认使用的相机模型是 SIMPLE_PINHOLE，fx=fy，cx,cy是主点坐标
        fx = cam.params[0] #焦距，单位像素
        cx = cam.params[1] #主点坐标，单位像素
        cy = cam.params[2]

        # 构建相机内参矩阵K
        self.K = np.array([[fx, 0, cx],
                          [0, fx, cy],
                          [0, 0, 1]], dtype=np.float32)

        print(f"相机内参: fx={fx}, cx={cx}, cy={cy}")
        print(f"地图加载完成: {len(self.point3d_to_desc)} 个带描述子的 3D 点")
    
    def get_camera_intrinsics(self):
        return self.K
    
    def match_3d_points(self, pts, desc):
        """匹配 3D 点（用 Annoy 加速 + Lowe's ratio test 过滤歧义匹配）
        pts: (3, N) 特征点坐标合集 (x, y, 置信度分数)，N 是特征点数量
        desc: (256, N) 描述子矩阵

        Lowe's ratio test: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform#Matching
        如果最佳匹配和次佳匹配差不多像，那宁愿放弃这次匹配，因为这说明说明这个特征太“平庸”（比如来自一面白墙或者重复的纹理）。

        """
        img_points = []
        world_points = []

        # 遍历每个 2D 特征点的描述子，找最近的 2 个 3D 点候选
        for i in range(desc.shape[1]): # desc.shape[1] 是特征点总数 N
            query_desc = desc[:, i]

            # 用 Annoy 找最近的 2 个 3D 点候选，计算 Lowe's ratio test
            nearest_indices = self.annoy_index.get_nns_by_vector(query_desc, n=2)
            # 返回annoy内部编号idx

            if len(nearest_indices) < 2: # 不足两个 跳过
                continue

            best_sim = -1 # 初始化变量
            second_best_sim = -1
            best_pid = None

            # 对候选重新计算点积
            for idx in nearest_indices: # 遍历这两个候选
                pid = self.pid_to_idx[idx] #转换：idx -> 3D点ID
                # 计算相似度（点积）来衡量描述子之间的相似性
                pdesc = self.point3d_to_desc[pid]
                sim = np.dot(query_desc, pdesc)
                # 如果当前相似度比之前记录的最高分还高，就更新“最佳”和“次佳”
                if sim > best_sim:
                    second_best_sim = best_sim
                    best_sim = sim
                    best_pid = pid

            # 洛氏检验Lowe's ratio test: 最佳与次佳比值要足够大（>1.5），说明最佳明确优于次佳
            # 同时最佳相似度也要足够高（>0.65）
            if best_sim > 0.65 and (second_best_sim < 1e-6 or best_sim / second_best_sim > 1.5):
                img_points.append([pts[0, i], pts[1, i]]) # 把当前新图片上的 2D 像素坐标 (x, y) 存进 img_points
                world_points.append(self.point3d_coords[best_pid]) # 通过 best_pid 从 point3d_coords 字典里取出对应的 3D 世界坐标 (X, Y, Z)，存进 world_points

        # 把列表img_points (M, 2) 和 world_points (M, 3)转换成numpy数组 并返回给调用者，M 是匹配成功的点对数量
        return np.array(img_points, dtype=np.float32), np.array(world_points, dtype=np.float32) 
        # cv2.solvePnPRansac() 要求输入的点集是三维坐标集

# ==================== pnp_solver.py ====================

class PnPSolver:
    """PnP 解算器

    传入形参：
    K: 相机内参矩阵(描述了 3D 点是如何投影到 2D 平面的缩放和位移关系)
    
    成员变量：
    K: 相机内参矩阵

    接口：
    solve(img_points, world_points): 给定 2D-3D 点对，返回相机位姿（旋转向量 rvec，平移向量 tvec）和内点数量
    draw_pose(frame, rvec, tvec): 在图像上画出相机坐标轴，便于可视化相机位姿

    """
    
    def __init__(self, K):
        self.K = K # 包含了焦距和主点坐标
    
    def solve(self, img_points, world_points):
        """解算相机位姿"""
        print(f"PnP 输入: {len(img_points)} 个点")  # ← this line is for debugging

        if len(img_points) < 4: # PnP 至少需要 4 组不共面的 2D-3D 点对, 否则数学上无解
            print("⚠️ 点数不足，无法解算 PnP")
            return None
        
        # ========== 在这里加打印样本点调试 ==========
        print("=== PnP 输入样本 ===")
        print("第一个 3D 点:", world_points[0])
        print("第一个 2D 点:", img_points[0])
        print("相机内参 K:", self.K)
        print("====================")
        # ==========================================

        '''
        RANSAC 随机抽样一致性算法（Random Sample Consensus）
        每一次循环：随机抽4个点对，解算相机位姿，计算所有点对的重投影误差，统计内点数量。
        循环2000次，最终选择内点数量最多的解作为最终结果。

        cv2.solvePnPRansac()返回值：
        success(bool): 是否成功解算
        rvec(numpy.ndarray): Rodrigues旋转向量(3, 1)或(3,)，相机姿态的旋转部分，述相机在世界坐标系中的朝向
        tvec(numpy.ndarray): 平移向量(3, 1)或(3,)，相机姿态的平移部分，描述相机在世界坐标系中的位置，单位与3D点坐标一致
        inliers(numpy.ndarray): 内点的索引列表(N, 1)，表示哪些点对被认为是内点，参与了最终的位姿解算。长度就是内点数量（RANSAC接受点数）。

        '''

        # 1. 先用 RANSAC 尝试
        # 匹配质量提升后，可以用更严格的参数
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_points, img_points, self.K, None, # none: 没有畸变参数
            iterationsCount=2000,
            reprojectionError=15.0, #允许15个像素的重投影误差（内点判定的阈值）
            confidence=0.9 # 要求90%的置信度，意味着希望最终的解至少有90%的概率是正确的
        )
        
        if success:
            print(f"✅ RANSAC 成功，内点数量: {len(inliers)}")
            return {
                'tvec': tvec.ravel(),
                'rvec': rvec.ravel(),
                'inliers': len(inliers)
                # 问题：此处只返回了len(inliers)，而不是inliers本身.
                # 如果做匹配质量的热力图分析，可能需要返回inliers数组本身。
            }
        
        # 2. RANSAC 失败，用普通 solvePnP 兜底
        print("⚠️ RANSAC 失败，尝试普通 solvePnP...")
        
        # 尝试不同参数
        for method in [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP]:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    world_points, img_points, self.K, None, 
                    flags=method
                )
                if success:
                    print(f"成功: 用 method={method}")
                    return {
                        'tvec': tvec.ravel(),
                        'rvec': rvec.ravel(),
                        'inliers': len(img_points)
                    }
            except:
                continue
        
        print("所有 PnP 方法都失败")
        return None
    
    def draw_pose(self, frame, rvec, tvec, offset=0, axis_length=0.4):
        """在图像上画相机坐标轴
        offset: 颜色偏移，0=红绿蓝，3=青品黄
        axis_length: 轴的长度（米），默认 0.1，改大可使轴更明显
        """
        axis = np.float32([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.K, None)
        imgpts = imgpts.reshape(-1, 2).astype(int)

        # cx, cy = int(self.K[0,2]), int(self.K[1,2]) #主点（光心），不是世界原点的投影！
       
        origin = tuple(imgpts[0])
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        if offset:
            colors = [(255,255,0), (0,255,255), (255,0,255)]
        for i, pt in enumerate(imgpts[1:]):
            cv2.line(frame, origin, tuple(pt), colors[i], 3)

# ==================== feature_extractor.py ====================

class FeatureExtractor:
    """特征提取器

    形参：
    weights_path: SuperPoint 预训练权重路径（.pth 文件）
    
    成员变量：
    fe: SuperPointFrontend 实例对象，用于提取特征点和描述子
        NMS 距离 = 4
        置信度阈值 = 0.015
    
    成员函数：
    extract(image): 给定一张图像，返回特征点坐标、描述子和热力图

    
    """
    
    def __init__(self, weights_path='superpoint_v1.pth'):
        self.fe = SuperPointFrontend(weights_path, nms_dist=4, 
                                       conf_thresh=0.015, nn_thresh=0.7)
    
    def extract(self, image):
        """提取特征点和描述子"""
        print("开始特征提取...")  

        if len(image.shape) == 3: # 检测传入图片image是否是彩色图像（3通道）
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 转为灰度图像

        # 缩放到最大 1024 像素（加速+省内存）内存防爆
        h, w = image.shape
        max_size = 1024
        scale = 1.0  # 记录缩放比例，初始化为 1.0（不缩放）
        if max(h, w) > max_size: # 如果图像的最大边长超过 1024，就按比例缩小
            scale = max_size / max(h, w)
            new_w = int(w * scale) # width
            new_h = int(h * scale) # height
            image = cv2.resize(image, (new_w, new_h))
        
        image = image.astype('float32') / 255.0
        pts, desc, heatmap = self.fe.run(image)
        #claude 修改：之前的代码里 pts 和 desc 是在缩放后的图像上提取的，所以需要把它们缩放回原图坐标系下，才能和地图上的 3D 点匹配。
        # 把特征点坐标缩放回原图分辨率
        if scale != 1.0:
            pts[:2, :] /= scale  # pts 是 (3, N)，前两行是 x, y

        return pts, desc, heatmap # 返回特征点坐标 (3, N)，描述子 (256, N)，热力图 (H, W)（并未使用）

# ==================== visualizer.py ====================

class Visualizer:
    """可视化工具"""
    
    @staticmethod # 静态方法装饰器，不需要实例化对象就可以调用
    def draw_text(frame, text, position, color=(0,255,0)):
        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    @staticmethod
    def draw_keypoints(frame, pts):
        for i in range(pts.shape[1]):
            x, y = int(pts[0, i]), int(pts[1, i])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


# ==================== localizer.py ====================

class Localizer:
    """定位器（整合各模块）
    形参：
    map_path: 3D地图路径(reconstruction文件夹)
    desc_path: 描述子路径
    weights_path: SuperPoint 预训练权重路径
    
    成员变量：
    map: MapManager 实例对象，管理 3D 地图和描述子
    solver: PnPSolver 实例对象，解算相机位姿
    extractor: FeatureExtractor 实例对象，提取特征点和描述子
    visualizer: Visualizer 实例对象，可视化工具
    
    成员函数：
    _process_frame(frame): 处理单帧图像，提取特征点，匹配 3D 点，解算位姿    
    process_image(image_path): 处理单张图片，保存结果图像
    run_camera(camera_id): 实时摄像头定位
    process_folder(folder_path, output_subdir): 批量处理文件夹中的所有图片
    """
    
    def __init__(self, map_path, desc_path, weights_path='superpoint_v1.pth'):
        self.map = MapManager(map_path, desc_path)
        self.solver = PnPSolver(self.map.get_camera_intrinsics()) # solvePnPRansac() 参数：相机内参矩阵 K
        self.extractor = FeatureExtractor(weights_path)
        self.visualizer = Visualizer()

    def _process_frame(self, frame): # frame: BGR 彩色图像
        """处理单帧图像，提取特征点，匹配 3D 点，解算位姿"""

        print("进入 _process_frame")  

        pts, desc, _ = self.extractor.extract(frame) # 提取特征点 描述子
        print(f"特征点数量: {pts.shape[1]}")  
        
        if pts.shape[1] < 10:
            print("特征点不足")
            return None  # 少于10个特征点就放弃
        
        img_points, world_points = self.map.match_3d_points(pts, desc)
        print(f"匹配到的 3D 点数量: {len(img_points)}")  
        
        if len(img_points) < 4:
            print("匹配点不足")
            return None  # 少于4个匹配点无法解算
        
        print("开始 PnP 解算...")  
        result = self.solver.solve(img_points, world_points) # 输入点对，返回rvec, tvec, len(inliers)（位姿）
        print("PnP 解算完成")      

        # 手动释放内存
        del pts, desc
        gc.collect()

        return result
    

    # ==================== 不同处理方式 ===================

    def process_image(self, image_path):
        """处理单张图片"""
        print(f"开始处理图片: {image_path}")  

        frame = cv2.imread(image_path)  # 用 OpenCV 读取硬盘上的图片。如果路径不对或文件损坏，frame 会是 None，直接退出
        if frame is None:
            print(f"无法读取图片: {image_path}")
            return None
        
        print("图片读取成功")  
        result = self._process_frame(frame) # result 包含了 rvec、tvec 和内点数量（如果解算成功的话）
        
        if result:
            # 生成输出文件名（在原图名后加 _pose）
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            out_name = f"{base_name}_pose.jpg"
            
            # 1. 画pnp坐标轴
            self.solver.draw_pose(frame, result['rvec'], result['tvec'])
            cv2.imwrite(out_name, frame)

            # =========== 调试代码：画COLMAP真值轴 ===========
            img_name = os.path.basename(image_path)
            for img in self.map.reconstruction.images.values():
                if img.name == img_name:

                    pose = img.cam_from_world()
    
                    # 获取四元数 (x, y, z, w)
                    quat = pose.rotation.quat  # [x, y, z, w]
                    
                    # 四元数转旋转矩阵
                    r = R.from_quat(quat)
                    R_true = r.as_matrix()
                    
                    # 转 Rodrigues 向量
                    rvec_true, _ = cv2.Rodrigues(R_true)
                    
                    # 平移向量
                    t_true = pose.translation

                    print("=== 位姿对比 ===")
                    print(f"真值位置: {t_true}")
                    print(f"估计位置: {result['tvec']}")
                    print(f"位置误差: {np.linalg.norm(t_true - result['tvec']):.2f} 米")
                        
                    # 画真值轴
                    self.solver.draw_pose(frame, rvec_true, t_true, offset=1)
                    break
            
            cv2.imwrite('pose_comparison.jpg', frame)
            print("对比图保存为 pose_comparison.jpg")
            # ==============================================

        
            print(f"相机位置: {result['tvec']}")
            print(f"结果保存为: {out_name}")



        return result
    

    def run_camera(self, camera_id=0):
        """实时摄像头定位"""
        cap = cv2.VideoCapture(camera_id)
        print("按 q 退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self._process_frame(frame)
            
            if result:
                self.visualizer.draw_text(frame, f"X: {result['tvec'][0]:.2f}m", (10, 30))
                self.visualizer.draw_text(frame, f"Y: {result['tvec'][1]:.2f}m", (10, 60))
                self.visualizer.draw_text(frame, f"Z: {result['tvec'][2]:.2f}m", (10, 90))
                self.solver.draw_pose(frame, result['rvec'], result['tvec'])
            else:
                self.visualizer.draw_text(frame, "Not enough matches", (10, 30), (0,0,255))
            
            cv2.imshow('PnP Localization', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def process_folder(self, folder_path, output_subdir='pnp_results'):
        """
        批量处理文件夹中的所有图片
        folder_path: 图片文件夹路径
        output_subdir: 结果保存的子文件夹名
        """
        # 查找所有图片
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if not image_files:
            print(f"文件夹 {folder_path} 中没有找到图片")
            return []
                
        # 创建输出文件夹
        if output_subdir is None:
            folder_name = os.path.basename(os.path.normpath(folder_path))
            output_subdir = f"{folder_name}_pnp"
        
        output_dir = os.path.join(folder_path, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"找到 {len(image_files)} 张图片")
        print(f"结果保存到: {output_dir}")
        
        results = []
        for img_path in image_files:
            print(f"\n处理: {os.path.basename(img_path)}")
            
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"  无法读取，跳过")
                continue
            
            result = self._process_frame(frame)
            
            if result:
                # 不加时间戳，直接覆盖
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                out_name = f"{base_name}_pose.jpg"
                out_path = os.path.join(output_dir, out_name)
                
                self.solver.draw_pose(frame, result['rvec'], result['tvec'])
                cv2.imwrite(out_path, frame)
                
                result['image'] = img_path
                result['output'] = out_path
                results.append(result)
                
                print(f"  ✅ 相机位置: {result['tvec']}")
                print(f"  📁 保存: {out_path}")
            else:
                print(f"  ❌ 定位失败")
                results.append({'image': img_path, 'success': False})

            
        print(f"\n完成: {len([r for r in results if r.get('success', True)])}/{len(image_files)} 张成功")
        return results


# ==================== main.py ====================
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PnP Localization Demo')
    parser.add_argument('input', type=str, help='Image file path or "camera"')
    parser.add_argument('--map', default=r'D:\COLMAP\project1\hloc_output\reconstruction')
    parser.add_argument('--desc', default='point3d_to_desc.npy')
    
    args = parser.parse_args()
    
    localizer = Localizer(args.map, args.desc)
    
    if args.input == 'camera':
        localizer.run_camera()
    elif os.path.isdir(args.input):
        localizer.process_folder(args.input)
    else:
        localizer.process_image(args.input)

