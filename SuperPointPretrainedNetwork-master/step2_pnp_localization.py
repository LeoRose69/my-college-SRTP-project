import pycolmap
import numpy as np
import cv2
from demo_superpoint import SuperPointFrontend

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
    
    def draw_pose(self, frame, rvec, tvec):
        """在图像上画相机坐标轴"""
        axis = np.float32([[0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.K, None)
        
        cx, cy = int(self.K[0,2]), int(self.K[1,2])
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        for i, pt in enumerate(imgpts):
            pt = tuple(pt.ravel().astype(int))
            cv2.line(frame, (cx, cy), pt, colors[i], 3)


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
        image = image.astype('float32') / 255.0
        return self.fe.run(image)


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
            print(f"相机位置: {result['tvec']}")
            self.solver.draw_pose(frame, result['rvec'], result['tvec'])
            cv2.imwrite('pose_result.jpg', frame)
            print("结果保存为 pose_result.jpg")
        
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


# ==================== main.py ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PnP Localization Demo')
    parser.add_argument('input', type=str, help='Image file path or "camera"')
    parser.add_argument('--map', default=r'D:\COLMAP\project1\hloc_output\reconstruction')
    parser.add_argument('--desc', default='point3d_to_desc.npy')
    
    args = parser.parse_args()
    
    localizer = Localizer(args.map, args.desc)
    
    if args.input == 'camera':
        localizer.run_camera()
    else:
        localizer.process_image(args.input)


