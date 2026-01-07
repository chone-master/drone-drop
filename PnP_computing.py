import cv2
import cv2.aruco as aruco
import numpy as np
import math


def get_dji_camera_matrix(width, height, dfov_deg=84.0):
    """
    根据 DJI 相机的 DFOV (通常 84度) 和图像分辨率，计算内参矩阵
    """
    # 1. 计算对角线像素长度
    diagonal_pixels = math.sqrt(width ** 2 + height ** 2)

    # 2. 计算焦距 (像素单位)
    # formula: f_pixels = (diagonal_pixels / 2) / tan(DFOV / 2)
    fov_rad = math.radians(dfov_deg)
    focal_length_px = (diagonal_pixels / 2) / math.tan(fov_rad / 2)

    # 3. 构造矩阵
    # [fx,  0, cx]
    # [ 0, fy, cy]
    # [ 0,  0,  1]
    center_x = width / 2
    center_y = height / 2

    # 假设像素是正方形，fx = fy
    camera_matrix = np.array([
        [focal_length_px, 0, center_x],
        [0, focal_length_px, center_y],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((5, 1))  # 大疆推流通常已经做过畸变矫正，这里设为0即可
    return camera_matrix, dist_coeffs


def main():
    # ================= 配置区域 =================
    image_path = "tag3.png"  # 你的测试图片

    # 真实物理尺寸 (单位：米)
    TAG_SIZE_BIG = 0.60  # 用户指定：60cm
    TAG_SIZE_SMALL = 0.12  # 估算：建议实测一下内部小码的边长

    # 你的推流分辨率 (很重要！Orin 收到的是多少分辨率就填多少)
    STREAM_W = 1400
    STREAM_H = 1327
    # ===========================================

    # 1. 生成相机内参
    camera_matrix, dist_coeffs = get_dji_camera_matrix(STREAM_W, STREAM_H)
    print(f"📷 相机内参计算完毕 (fx={camera_matrix[0][0]:.1f})")

    # 2. 准备检测器
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # 读取图片 (实际应用中这里是 cap.read())
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ 无法读取图片")
        return

    # 如果图片尺寸和设定的流分辨率不一致，强行缩放模拟真实情况
    if frame.shape[1] != STREAM_W:
        frame = cv2.resize(frame, (STREAM_W, STREAM_H))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        print(f"✨ 检测到 {len(ids)} 个 Tag")

        # 3. 定义 Tag 的 3D 坐标点 (以 Tag 中心为 0,0,0)
        def get_obj_points(size):
            half = size / 2
            return np.array([
                [-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]
            ], dtype=np.float32)

        for i in range(len(ids)):
            current_id = ids[i][0]

            # 策略：根据 ID 切换尺寸
            if current_id in [571, 576]:  # 小码
                obj_points = get_obj_points(TAG_SIZE_SMALL)
                tag_name = "小码 (降落)"
            else:  # 大码
                obj_points = get_obj_points(TAG_SIZE_BIG)
                tag_name = "大码 (定位)"

            # PnP 解算
            # solvePnP 接收：3D点, 2D角点, 内参, 畸变
            ret, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)

            # --- 这一步最关键：坐标系转换 ---
            # tvec 里的数据是【相机坐标系】：
            # x_cam: 右为正
            # y_cam: 下为正
            # z_cam: 前为正 (即高度)

            x_cam = tvec[0][0]
            y_cam = tvec[1][0]
            z_cam = tvec[2][0]  # 这就是高度

            # 转换为【无人机机体坐标系】(Body Frame)
            # 假设相机垂直朝下安装，机头朝向画面上方：
            # 1. 图像的右 (x_cam) -> 无人机的右 (Roll) -> 需要向右飞
            # 2. 图像的下 (y_cam) -> 无人机的后 (Pitch) -> 需要向后飞

            # 偏差 (Error) = 目标 - 当前
            # 目标是 (0,0)，所以 Error = 0 - pos
            err_roll = -x_cam  # 偏差为负，表示要向左修；偏差为正，向右修
            err_pitch = y_cam  # 注意方向：如果 Tag 在画面下方 (y_cam>0)，说明飞机太靠前了，需要向后飞

            print(f"--------------------------------")
            print(f"🎯 目标: {tag_name} [ID: {current_id}]")
            print(f"   📏 高度 (Z): {z_cam:.2f} m")
            print(f"   ↔️  横向偏差 (Cam X): {x_cam:.2f} m")
            print(f"   ↕️  纵向偏差 (Cam Y): {y_cam:.2f} m")
            print(f"   👉 指令预测: 向{'左' if err_roll > 0 else '右'}飞 {abs(err_roll):.2f}m, "
                  f"向{'前' if err_pitch > 0 else '后'}飞 {abs(err_pitch):.2f}m")

            # 画轴
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.3)

        # 显示
        cv2.imshow("Landing View", cv2.resize(frame, (960, 540)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()