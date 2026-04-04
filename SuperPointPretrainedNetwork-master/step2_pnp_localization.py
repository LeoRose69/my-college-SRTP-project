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



# ==================== map_manager.py ====================

class MapManager:
    """管理 3D 地图"""
    
    def __init__(self, map_path, desc_path):
        self.reconstruction = pycolmap.Reconstruction()
        self.reconstruction.read_binary(map_path)
        
        # 3D 点坐标
        self.point3d_coords = {}
        for pid, p3d in self.reconstruction.points3D.items():
            self.point3d_coords[pid] = p3d.xyz
        
        # 描述子映射
        self.point3d_to_desc = np.load(desc_path, allow_pickle=True).item()
        
        # 相机内参
        cam = next(iter(self.reconstruction.cameras.values()))
        fx = cam.params[0]
        cx = cam.params[1]
        cy = cam.params[2]
        self.K = np.array([[fx, 0, cx],
                          [0, fx, cy],
                          [0, 0, 1]], dtype=np.float32)
        
        print(f"相机内参: fx={fx}, cx={cx}, cy={cy}")
        print(f"地图加载完成: {len(self.point3d_to_desc)} 个带描述子的 3D 点")
    
    def get_camera_intrinsics(self):
        return self.K
    
    def match_3d_points(self, pts, desc):  # ← 加上 pts
        """匹配 3D 点
        pts: (3, N) 特征点坐标
        desc: (256, N) 描述子
        """
        img_points = []
        world_points = []
        
        for i in range(desc.shape[1]):
            query_desc = desc[:, i]
            best_sim = -1
            best_pid = None
            
            # 暴力匹配，复杂度O(n*m)，n是图像特征点数量，m是地图3D点数量。对于小规模地图和实时性要求不高的场景，这种方法是可行的。
            # 如果地图点数量非常大，可以考虑使用 KD 树或其他近似最近邻算法来加速匹配。
            for pid, pdesc in self.point3d_to_desc.items():
                sim = np.dot(query_desc, pdesc)
                if sim > best_sim:
                    best_sim = sim
                    best_pid = pid
            
            if best_sim > 0.7:
                img_points.append([pts[0, i], pts[1, i]])  # ← 现在 pts 有定义了
                world_points.append(self.point3d_coords[best_pid])
        
        return np.array(img_points, dtype=np.float32), np.array(world_points, dtype=np.float32)

# ==================== pnp_solver.py ====================

class PnPSolver:
    """PnP 解算器"""
    
    def __init__(self, K):
        self.K = K
    
    def solve(self, img_points, world_points):
        """解算相机位姿"""
        print(f"PnP 输入: {len(img_points)} 个点")  # ← 加这行

        if len(img_points) < 4:
            return None
        
        # ========== 在这里加打印样本点调试 ==========
        print("=== PnP 输入样本 ===")
        print("第一个 3D 点:", world_points[0])
        print("第一个 2D 点:", img_points[0])
        print("相机内参 K:", self.K)
        print("====================")
        # ==========================================

        # 1. 先用 RANSAC 尝试
        # RANSAC 1000 次迭代(效率低下)，重投影误差阈值 10 像素，置信度 95%
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_points, img_points, self.K, None,
            iterationsCount=1000,
            reprojectionError=10.0,
            confidence=0.95
        )
        
        if success:
            print(f"✅ RANSAC 成功，内点数量: {len(inliers)}")
            return {
                'tvec': tvec.ravel(),
                'rvec': rvec.ravel(),
                'inliers': len(inliers)
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
    """特征提取器"""
    
    def __init__(self, weights_path='superpoint_v1.pth'):
        self.fe = SuperPointFrontend(weights_path, nms_dist=4, 
                                       conf_thresh=0.015, nn_thresh=0.7)
    
    def extract(self, image):
        """提取特征点和描述子"""
        print("开始特征提取...")  # ← 加这行

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 缩放到最大 1024 像素（加速+省内存）内存防爆
        h, w = image.shape
        max_size = 1024
        scale = 1.0  # 记录缩放比例
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        image = image.astype('float32') / 255.0
        pts, desc, heatmap = self.fe.run(image)
        #claude 修改：之前的代码里 pts 和 desc 是在缩放后的图像上提取的，所以需要把它们缩放回原图坐标系下，才能和地图上的 3D 点匹配。
        # 把特征点坐标缩放回原图分辨率
        if scale != 1.0:
            pts[:2, :] /= scale  # pts 是 (3, N)，前两行是 x, y

        return pts, desc, heatmap

# ==================== visualizer.py ====================

class Visualizer:
    """可视化工具"""
    
    @staticmethod
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
    """定位器（整合各模块）"""
    
    def __init__(self, map_path, desc_path, weights_path='superpoint_v1.pth'):
        self.map = MapManager(map_path, desc_path)
        self.solver = PnPSolver(self.map.get_camera_intrinsics())
        self.extractor = FeatureExtractor(weights_path)
        self.visualizer = Visualizer()
    
    def _process_frame(self, frame):
        print("进入 _process_frame")  # ← 加这行

        pts, desc, _ = self.extractor.extract(frame)
        print(f"特征点数量: {pts.shape[1]}")  # ← 加这行
        
        if pts.shape[1] < 10:
            print("特征点不足")
            return None  # 少于10个特征点就放弃
        
        img_points, world_points = self.map.match_3d_points(pts, desc)
        print(f"匹配到的 3D 点数量: {len(img_points)}")  # ← 加这行
        
        if len(img_points) < 4:
            print("匹配点不足")
            return None  # 少于4个匹配点无法解算
        
        print("开始 PnP 解算...")  # ← 加这行
        result = self.solver.solve(img_points, world_points)
        print("PnP 解算完成")      # ← 加这行

        # 手动释放内存
        del pts, desc
        gc.collect()

        return result
    

    def process_image(self, image_path):
        """处理单张图片"""
        print(f"开始处理图片: {image_path}")  # ← 加这行

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图片: {image_path}")
            return None
        
        print("图片读取成功")  # ← 加这行
        result = self._process_frame(frame)
        
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



        ''' # 加时间戳的版本
        if result:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = datetime.now().strftime('%Y%f_%H%M%S')
            out_name = f"{base_name}_pose_{timestamp}.jpg"
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, out_name)
            else:
                out_path = out_name
            
            self.solver.draw_pose(frame, result['rvec'], result['tvec'])
            cv2.imwrite(out_path, frame)
            
            print(f"相机位置: {result['tvec']}")
            print(f"结果保存为: {out_path}")

        '''

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

            ''' # 加时间戳版本
             if result:
                # 保存结果
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                timestamp = datetime.now().strftime('%H%M%S')
                out_name = f"{base_name}_pose_{timestamp}.jpg"
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
                
            '''

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

