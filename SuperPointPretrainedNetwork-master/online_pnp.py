#first attempt(chessboard pnp)
'''which was an absolute failure'''

import cv2
import numpy as np
from demo_superpoint import SuperPointFrontend

# 1. 加载 SuperPoint
fe = SuperPointFrontend('superpoint_v1.pth', 4, 0.015, 0.7)

# 2. 棋盘格参数（9×6，格子边长3cm）
objp = np.zeros((54, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objp = objp * 0.03

# 3. 相机内参（640×480的估算值）
K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)

# 4. 打开摄像头
cap = cv2.VideoCapture(0)

#调试代码
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 尝试检测棋盘格
    ret_c, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    if ret_c:
        print("✅ 检测到棋盘格！")
        # 画出来
        cv2.drawChessboardCorners(frame, (9,6), corners, ret_c)
        cv2.putText(frame, "CHESSBOARD DETECTED!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No chessboard detected...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Chessboard Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype('float32') / 255.0
    
    # 5. SuperPoint 找特征点
    pts, desc, _ = fe.run(gray_float)
    
    # 6. 用 OpenCV 找棋盘格（更可靠）
    ret_c, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    if ret_c:
        img_points = corners.reshape(-1, 2)
        world_points = objp
        
        # 7. PnP 解算
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            world_points.astype(np.float32),
            img_points.astype(np.float32),
            K, None
        )
        
        if success:
            # ========== 1. 显示坐标（包含X Y Z） ==========
            x, y, z = tvec[0,0], tvec[1,0], tvec[2,0]
            
            # 用不同颜色显示XYZ
            cv2.putText(frame, f"X: {x:.3f} m", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # 红色
            cv2.putText(frame, f"Y: {y:.3f} m", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # 绿色
            cv2.putText(frame, f"Z: {z:.3f} m", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)  # 蓝色
            
            # ========== 2. 画相机坐标轴（3D箭头） ==========
            # 定义三个轴的方向（长度0.1米）
            axis = np.float32([[0.1, 0, 0],   # X轴终点
                               [0, 0.1, 0],   # Y轴终点
                               [0, 0, 0.1]]).reshape(-1, 3)  # Z轴终点
            
            # 投影到图像平面
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, None)
            
            # 图像中心（相机的光心）
            cx, cy = int(K[0,2]), int(K[1,2])
            
            # 画三个轴（不同颜色）
            # X轴 - 红色
            x_end = tuple(imgpts[0].ravel().astype(int))
            cv2.line(frame, (cx, cy), x_end, (0, 0, 255), 3)
            # Y轴 - 绿色
            y_end = tuple(imgpts[1].ravel().astype(int))
            cv2.line(frame, (cx, cy), y_end, (0, 255, 0), 3)
            # Z轴 - 蓝色
            z_end = tuple(imgpts[2].ravel().astype(int))
            cv2.line(frame, (cx, cy), z_end, (255, 0, 0), 3)
            
            # 在轴端点画小圆点
            cv2.circle(frame, x_end, 4, (0, 0, 255), -1)
            cv2.circle(frame, y_end, 4, (0, 255, 0), -1)
            cv2.circle(frame, z_end, 4, (255, 0, 0), -1)
            
            # 可选：在棋盘格上画出检测到的角点
            cv2.drawChessboardCorners(frame, (9,6), corners, ret_c)

        # 显示提示信息
    cv2.putText(frame, "Show chessboard to camera", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Online PnP - Camera Pose Estimation', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # 按s键截图
        cv2.imwrite('screenshot.png', frame)
        print("✅ 截图已保存为 screenshot.png")
'''

cap.release()
cv2.destroyAllWindows()


