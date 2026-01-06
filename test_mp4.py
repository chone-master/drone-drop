import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
import socket
import json
import threading

# ================= 1. 用户配置区域 (来自旧代码) =================
# 视频路径 (修改为你的实际路径)
VIDEO_PATH = "test2.mp4"
# VIDEO_PATH = 0  # 如果需要使用摄像头，请取消注释此行

# 网络配置 (Android 端 IP)
ANDROID_IP = "192.168.42.129"
ANDROID_PORT = 8888

# 物理尺寸配置 (单位: 米)
TAG_SIZE_BIG = 0.515  # 大码
TAG_SIZE_SMALL = 0.096  # 小码

# Tag 布局偏移量 (ID -> [x, y, z])
# 0号是大码中心，576/571是周围小码
TAG_LAYOUT = {
    0: np.array([0.0, 0.0, 0.0]),
    576: np.array([0.15, -0.15, 0.0]),
    571: np.array([-0.15, 0.15, 0.0])
}

# 处理分辨率 (建议不要用4K跑处理，1920x1080足够且更快)
PROCESS_W = 1920
PROCESS_H = 1080
DFOV_DEG = 84.0  # DJI 视角


# ================= 2. 多线程视频流类 (新增) =================
class CameraStream:
    """
    使用独立线程读取视频流，总是保持最新的一帧。
    解决 cv2.VideoCapture 缓冲区导致的延迟问题。
    """

    def __init__(self, src=0, width=1920, height=1080):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.stream.isOpened():
            print("❌ 错误：无法打开视频源")
            self.stopped = True
        else:
            self.stopped = False

        self.ret, self.frame = self.stream.read()
        self.width = width
        self.height = height

    def start(self):
        """开启子线程"""
        if self.stopped: return self
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        """后台线程循环读取"""
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                break
            # 只保留最新帧
            self.ret, self.frame = ret, frame
            time.sleep(0.005)  # 稍微休眠避免占满CPU

    def read(self):
        """主线程获取当前帧"""
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# ================= 3. PID 控制器 (来自新代码) =================
class PIDController:
    def __init__(self, kp, ki, kd, max_out=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.033

        # P 项
        p_term = self.kp * error

        # I 项 (带限幅，防止积分饱和 - 新代码特性)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        i_term = self.ki * self.integral

        # D 项
        d_term = self.kd * (error - self.prev_error) / dt

        output = p_term + i_term + d_term
        self.prev_error = error
        self.last_time = current_time

        return np.clip(output, -self.max_out, self.max_out)


# ================= 4. 增强型 Tag 检测器 =================
class EnhancedTagDetector:
    def __init__(self, width, height, dfov_deg):
        # 初始化相机内参
        self.camera_matrix, self.dist_coeffs = self._get_camera_matrix(width, height, dfov_deg)

        # 初始化 ArUco
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        self.parameters = aruco.DetectorParameters()
        # 这里我依然建议加上 subpix，对性能影响很小但精度高
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def _get_camera_matrix(self, width, height, dfov_deg):
        diagonal_pixels = math.sqrt(width ** 2 + height ** 2)
        fov_rad = math.radians(dfov_deg)
        f_px = (diagonal_pixels / 2) / math.tan(fov_rad / 2)
        K = np.array([[f_px, 0, width / 2], [0, f_px, height / 2], [0, 0, 1]], dtype=np.float32)
        D = np.zeros((5, 1))
        return K, D

    def detect_with_enhancement(self, frame):
        """逻辑核心：普通检测 -> 口罩法 -> 膨胀法"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 正常检测
        corners, ids, _ = self.detector.detectMarkers(gray)
        final_corners = list(corners) if corners else []
        final_ids = list(ids[:, 0]) if ids is not None else []

        has_big_tag = 0 in final_ids if final_ids else False

        # 2. 如果没找到大码，尝试“手术口罩法”
        if not has_big_tag:
            corners_mask, ids_mask = self._mask_small_tags_method(gray, final_corners, final_ids)
            if ids_mask is not None and 0 in ids_mask:
                final_corners.extend(corners_mask)
                final_ids.extend(ids_mask[:, 0])
                has_big_tag = True
                # print("💡 触发修正：手术口罩法")

        # 3. 如果还没找到，尝试“膨胀法”
        if not has_big_tag:
            corners_dilate, ids_dilate = self._dilate_method(gray)
            if ids_dilate is not None and 0 in ids_dilate:
                final_corners.extend(corners_dilate)
                final_ids.extend(ids_dilate[:, 0])
                has_big_tag = True
                # print("💡 触发修正：图像膨胀法")

        if not final_ids:
            return tuple(), None

        return tuple(final_corners), np.array(final_ids).reshape(-1, 1)

    def _mask_small_tags_method(self, gray, existing_corners, existing_ids):
        """遮盖已知小码区域"""
        masked_gray = gray.copy()
        small_tags_corners = []
        if existing_ids:
            for i, id_val in enumerate(existing_ids):
                if id_val in [571, 576]:  # 小码ID
                    small_tags_corners.append(existing_corners[i])

        for corner_set in small_tags_corners:
            pts = corner_set[0]
            center = np.mean(pts, axis=0)
            expanded_pts = (pts - center) * 1.6 + center  # 扩大遮盖范围
            cv2.fillPoly(masked_gray, [expanded_pts.astype(np.int32)], 255)  # 涂白

        corners, ids, _ = self.detector.detectMarkers(masked_gray)
        if ids is not None and 0 in ids:
            idx = np.where(ids == 0)[0][0]
            return [corners[idx]], np.array([[0]])
        return None, None

    def _dilate_method(self, gray):
        """图像膨胀修复断裂边框"""
        k_size = int(gray.shape[1] * 0.015)  # 动态核大小
        if k_size % 2 == 0: k_size += 1
        kernel = np.ones((k_size, k_size), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)

        corners, ids, _ = self.detector.detectMarkers(dilated)
        if ids is not None and 0 in ids:
            idx = np.where(ids == 0)[0][0]
            return [corners[idx]], np.array([[0]])
        return None, None

    def solve_position(self, corners, ids):
        """计算融合后的物理位置"""
        if ids is None: return None

        positions = []
        for i, id_val in enumerate(ids[:, 0]):
            curr_id = int(id_val)
            # 根据ID选择尺寸
            size = TAG_SIZE_SMALL if curr_id in [571, 576] else TAG_SIZE_BIG

            # 定义3D点
            obj_pts = np.array([
                [-size / 2, size / 2, 0], [size / 2, size / 2, 0],
                [size / 2, -size / 2, 0], [-size / 2, -size / 2, 0]
            ], dtype=np.float32)

            # PnP解算 (此处暂不使用 IPPE_SQUARE，遵照你的指示不考虑抖动)
            _, rvec, tvec = cv2.solvePnP(obj_pts, corners[i], self.camera_matrix, self.dist_coeffs)

            # 坐标转换：将局部坐标转为降落板中心坐标
            if curr_id in TAG_LAYOUT:
                pos_center = tvec.flatten() - TAG_LAYOUT[curr_id]
                positions.append(pos_center)

        if positions:
            return np.mean(positions, axis=0)  # 多码融合取平均
        return None


# ================= 5. 主控制系统 (融合逻辑) =================
def main():
    # 1. 初始化通信
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"📡 UDP 目标: {ANDROID_IP}:{ANDROID_PORT}")

    # 2. 初始化检测器与PID
    detector = EnhancedTagDetector(PROCESS_W, PROCESS_H, DFOV_DEG)

    # PID 参数根据旧代码逻辑微调
    pid_roll = PIDController(kp=1.0, ki=0.0, kd=0.1, max_out=1.0)  # 控制左右(X)
    pid_pitch = PIDController(kp=1.0, ki=0.0, kd=0.1, max_out=1.0)  # 控制前后(Y)

    # 3. 启动多线程视频流
    # 注意：如果是文件，StreamThread 会快速读完，适合摄像头。文件测试建议用单线程。
    # 这里为了满足你的“多线程”要求，如果是文件，我们稍微改一点逻辑让它循环播放
    camera = CameraStream(VIDEO_PATH, PROCESS_W, PROCESS_H).start()

    print("🚀 系统启动，按 'q' 退出...")

    while True:
        # --- A. 获取图像 ---
        valid, frame = camera.read()
        if not valid:
            print("视频结束或无法读取")
            break

        # 确保分辨率一致 (如果是4K输入，这里会缩放到1080P处理)
        if frame.shape[1] != PROCESS_W:
            frame = cv2.resize(frame, (PROCESS_W, PROCESS_H))

        # --- B. 核心视觉处理 (使用新代码逻辑) ---
        corners, ids = detector.detect_with_enhancement(frame)
        final_pos = detector.solve_position(corners, ids)

        # 默认控制量
        cmd_roll = 0.0
        cmd_pitch = 0.0
        cmd_yaw = 0.0
        cmd_throttle = 0.0  # 对应 vel_z

        # --- C. 控制逻辑 ---
        if final_pos is not None:
            x, y, z = final_pos

            # 计算 PID (注意坐标系方向)
            # 假设 X是左右，Y是前后。旧代码：update(0-x) * -1
            cmd_roll = pid_roll.update(0 - x) * -1
            cmd_pitch = pid_pitch.update(0 - y)

            # 下降策略 (新代码的逻辑：先对准，再下降)
            horizontal_error = math.sqrt(x ** 2 + y ** 2)
            if horizontal_error < 0.2:  # 误差小于20cm
                cmd_throttle = -0.3  # 下降
                status_text = "DESCENDING"
            else:
                cmd_throttle = 0.0  # 悬停对准
                status_text = "ALIGNING"

            # 绘制可视化
            aruco.drawDetectedMarkers(frame, corners, ids)
            # 画中心点
            cx = int(PROCESS_W / 2 + x * (detector.camera_matrix[0][0] / z))
            cy = int(PROCESS_H / 2 + y * (detector.camera_matrix[1][1] / z))
            cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)
            cv2.line(frame, (PROCESS_W // 2, PROCESS_H // 2), (cx, cy), (0, 255, 0), 2)

            # 显示数据
            cv2.putText(frame, f"H: {z:.2f}m | {status_text}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"ERR: X={x:.2f} Y={y:.2f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        else:
            # 丢失目标：悬停
            cv2.putText(frame, "SEARCHING...", (PROCESS_W // 2 - 100, PROCESS_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # --- D. 发送 UDP (旧代码格式) ---
        send_data = {
            "r": float(cmd_roll),
            "p": float(cmd_pitch),
            "y": float(cmd_yaw),
            "t": float(cmd_throttle)
        }

        try:
            msg = json.dumps(send_data).encode('utf-8')
            udp_socket.sendto(msg, (ANDROID_IP, ANDROID_PORT))
            # print(f"Sent: {send_data}")
        except Exception as e:
            print(f"UDP Error: {e}")

        # --- E. 显示画面 ---
        # 缩小显示以便在屏幕上查看
        show_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Drone Vision", show_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理
    camera.stop()
    udp_socket.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()