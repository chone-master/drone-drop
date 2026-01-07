import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time
import socket
import json
import threading
import queue
import os

# ================= 1. 配置区域 =================

# --- RTSP 连接配置 ---
RTSP_USERNAME = "djiuser"
RTSP_PASSWORD = "123456"
RTSP_IP = "192.168.31.106"  # 遥控器 IP (M3E/M30 通常是这个)
RTSP_PORT = 8554
RTSP_PATH = "streaming/live/1"

# --- 安卓端通信配置 ---
ANDROID_IP = "192.168.31.106"
ANDROID_PORT = 8888

# --- 物理尺寸与布局 ---
TAG_SIZE_BIG = 0.515
TAG_SIZE_SMALL = 0.096

TAG_LAYOUT = {
    0: np.array([0.0, 0.0, 0.0]),
    576: np.array([0.075, 0.0015, 0.0]),
    571: np.array([-0.09, -0.052, 0.0])
}

# --- 分辨率设置 ---
# 输入流分辨率 (DJI M3E 通常是 4K 或 1080P)
# 我们不需要手动指定输入分辨率，OpenCV 会自动识别
# PROCESS_W/H 是算法处理时的分辨率，建议 1920x1080 以平衡速度与精度
PROCESS_W = 1920
PROCESS_H = 1080
# PROCESS_W = 3840
# PROCESS_H = 2160

DFOV_DEG = 84.0


# ================= 2. 线程工具类 =================

class RTSPStreamLoader:
    """
    RTSP 专用拉流线程：
    1. 自动断线重连
    2. 永远只保留最新一帧 (丢弃旧帧)
    """

    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.connect_success = False

        # 强制使用 UDP 传输以降低延迟
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

        print(f"📡 正在连接 RTSP: {rtsp_url} ...")
        self.connect()

    def connect(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.rtsp_url)
        # 缓冲区设置为1，尽可能减少积压
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            self.connect_success = True
            print("✅ RTSP 连接成功！")
        else:
            print("❌ RTSP 连接失败，将在后台重试...")

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(1)
                self.connect()
                continue

            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                # 读取失败（可能是断流）
                print("⚠️ RTSP 无数据，尝试重连...")
                time.sleep(0.5)
                self.connect()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.cap: self.cap.release()


class UDPSender:
    """
    UDP 发送线程：
    避免网络 IO 阻塞主线程的图像处理
    """

    def __init__(self, ip, port):
        self.target_addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.queue = queue.Queue(maxsize=1)  # 只存最新的一条
        self.stopped = False
        print(f"📡 UDP 发送服务启动 -> {ip}:{port}")

    def start(self):
        threading.Thread(target=self.run, args=(), daemon=True).start()
        return self

    def send_async(self, data):
        if self.stopped: return
        try:
            # 如果队列满，丢弃旧的，放入新的
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except:
                    pass
            self.queue.put_nowait(data)
        except:
            pass

    def run(self):
        while not self.stopped:
            try:
                data = self.queue.get(timeout=0.5)
                msg = json.dumps(data).encode('utf-8')
                self.sock.sendto(msg, self.target_addr)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"UDP Error: {e}")

    def stop(self):
        self.stopped = True
        self.sock.close()


# ================= 3. 算法工具类 =================

def apply_deadband(val):
    """死区控制：防止由于噪点导致无人机微小抖动"""
    MIN_MOVE_SPEED = 0.12  # 最小动作速度
    STOP_THRESHOLD = 0.05  # 死区阈值

    if abs(val) < STOP_THRESHOLD:
        return 0.0
    elif abs(val) < MIN_MOVE_SPEED:
        return math.copysign(MIN_MOVE_SPEED, val)
    else:
        return val


class PIDController:
    def __init__(self, kp, ki, kd, max_out=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        if dt <= 0: dt = 0.033

        p_term = self.kp * error
        d_term = self.kd * (error - self.prev_error) / dt

        self.integral += error * dt
        self.integral = np.clip(self.integral, -0.5, 0.5)
        i_term = self.ki * self.integral

        output = p_term + i_term + d_term
        self.prev_error = error
        return np.clip(output, -self.max_out, self.max_out)


class EnhancedTagDetector:
    """
    【保留增强检测】
    集成口罩法和膨胀法，保证在复杂环境下也能识别
    """

    def __init__(self, width, height, dfov_deg):
        self.camera_matrix, self.dist_coeffs = self._get_camera_matrix(width, height, dfov_deg)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        self.parameters = aruco.DetectorParameters()
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 常规检测
        corners, ids, _ = self.detector.detectMarkers(gray)
        final_corners = list(corners) if corners else []
        final_ids = list(ids[:, 0]) if ids is not None else []

        has_big_tag = 0 in final_ids if final_ids else False

        # 2. 口罩法
        if not has_big_tag:
            corners_mask, ids_mask = self._mask_small_tags_method(gray, final_corners, final_ids)
            if ids_mask is not None and 0 in ids_mask:
                final_corners.extend(corners_mask)
                final_ids.extend(ids_mask[:, 0])
                has_big_tag = True

        # 3. 膨胀法
        if not has_big_tag:
            corners_dilate, ids_dilate = self._dilate_method(gray)
            if ids_dilate is not None and 0 in ids_dilate:
                final_corners.extend(corners_dilate)
                final_ids.extend(ids_dilate[:, 0])

        if not final_ids: return tuple(), None
        return tuple(final_corners), np.array(final_ids).reshape(-1, 1)

    def _mask_small_tags_method(self, gray, existing_corners, existing_ids):
        masked_gray = gray.copy()
        small_tags_corners = []
        if existing_ids:
            for i, id_val in enumerate(existing_ids):
                if id_val in [571, 576]:
                    small_tags_corners.append(existing_corners[i])
        for corner_set in small_tags_corners:
            pts = corner_set[0]
            center = np.mean(pts, axis=0)
            expanded_pts = (pts - center) * 1.6 + center
            cv2.fillPoly(masked_gray, [expanded_pts.astype(np.int32)], 255)
        corners, ids, _ = self.detector.detectMarkers(masked_gray)
        if ids is not None and 0 in ids:
            return [corners[np.where(ids == 0)[0][0]]], np.array([[0]])
        return None, None

    def _dilate_method(self, gray):
        k_size = int(gray.shape[1] * 0.015)
        if k_size % 2 == 0: k_size += 1
        dilated = cv2.dilate(gray, np.ones((k_size, k_size), np.uint8), iterations=1)
        corners, ids, _ = self.detector.detectMarkers(dilated)
        if ids is not None and 0 in ids:
            return [corners[np.where(ids == 0)[0][0]]], np.array([[0]])
        return None, None

    def solve_position(self, corners, ids):
        if ids is None: return None
        positions = []
        for i, id_val in enumerate(ids[:, 0]):
            curr_id = int(id_val)
            size = TAG_SIZE_SMALL if curr_id in [571, 576] else TAG_SIZE_BIG
            obj_pts = np.array([[-size / 2, size / 2, 0], [size / 2, size / 2, 0],
                                [size / 2, -size / 2, 0], [-size / 2, -size / 2, 0]], dtype=np.float32)

            # 使用基础 solvePnP (不使用 IPPE 以避免抖动，或按需开启)
            _, rvec, tvec = cv2.solvePnP(obj_pts, corners[i], self.camera_matrix, self.dist_coeffs)
            if curr_id in TAG_LAYOUT:
                positions.append(tvec.flatten() - TAG_LAYOUT[curr_id])

        if positions: return np.mean(positions, axis=0)
        return None


# ================= 4. 主流程 =================
def main():
    # 1. 启动硬件线程
    rtsp_url = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}/{RTSP_PATH}"
    stream_loader = RTSPStreamLoader(rtsp_url).start()
    udp_sender = UDPSender(ANDROID_IP, ANDROID_PORT).start()

    # 2. 初始化增强型检测器
    detector = EnhancedTagDetector(PROCESS_W, PROCESS_H, DFOV_DEG)

    # 3. 初始化 PID
    pid_roll = PIDController(kp=1.0, ki=0.0, kd=0.1)
    pid_pitch = PIDController(kp=1.0, ki=0.0, kd=0.1)

    print("🚀 融合系统全速运行中... (等待视频流)")

    # 等待第一帧
    while stream_loader.get_frame() is None:
        time.sleep(0.1)

    last_time = time.time()
    # 【新增】用于记忆最后的高度
    last_valid_height = 10.0  # 初始值给大一点，防止误判
    # 【新增】盲降计时器（可选，防止无限盲降）
    blind_land_start_time = 0
    try:
        while True:
            # --- A. 获取最新帧 ---
            frame = stream_loader.get_frame()
            if frame is None: continue

            # 计算 dt
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # 缩放至 1080P 处理 (提升速度)
            if frame.shape[1] != PROCESS_W:
                frame = cv2.resize(frame, (PROCESS_W, PROCESS_H))

            # --- B. 增强视觉处理 ---
            corners, ids = detector.detect_with_enhancement(frame)
            final_pos = detector.solve_position(corners, ids)

            # --- C. PID 计算与死区控制 ---
            cmd_roll, cmd_pitch, vel_z = 0.0, 0.0, 0.0
            if final_pos is not None:
                x, y, z = final_pos
                # 【关键】时刻更新记忆
                last_valid_height = z
                blind_land_start_time = 0  # 重置盲降计时
                # 原始 PID 输出
                raw_roll = pid_roll.update(0 - x, dt=dt) * -1
                raw_pitch = pid_pitch.update(0 - y, dt=dt)

                # 应用死区 (防止微小抖动)
                cmd_roll = apply_deadband(raw_roll)
                cmd_pitch = apply_deadband(raw_pitch)

                # 下降逻辑
                horizontal_error = math.sqrt(x ** 2 + y ** 2)

                if z < 0.4:
                    vel_z = -0.15  # 强制慢速触地
                    # 在极低空，为了防止画面边缘畸变导致的误修，可以锁死水平控制
                    # cmd_r = 0.0
                    # cmd_p = 0.0
                elif horizontal_error < 0.3:
                    vel_z = max(-0.3, -z * 0.3)  # 比例下降
                else:
                    vel_z = 0.0

                # --- 绘制 UI (保留你喜欢的大字体风格) ---
                aruco.drawDetectedMarkers(frame, corners, ids)

                # 中心点与连线
                cx = int(PROCESS_W / 2 + x * (detector.camera_matrix[0][0] / z))
                cy = int(PROCESS_H / 2 + y * (detector.camera_matrix[1][1] / z))
                screen_cx, screen_cy = PROCESS_W // 2, PROCESS_H // 2

                cv2.line(frame, (screen_cx, screen_cy), (cx, cy), (0, 255, 0), 3)
                cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)

                # 文字信息
                info_text = [
                    f"H: {z:.2f}m",
                    f"Err: {horizontal_error:.2f}m",
                    f"R: {cmd_roll:.2f} P: {cmd_pitch:.2f}"
                ]
                for idx, text in enumerate(info_text):
                    cv2.putText(frame, text, (30, 80 + idx * 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                # 箭头指示
                arrow_len = 150
                end_x = int(screen_cx + cmd_roll * arrow_len)
                end_y = int(screen_cy - cmd_pitch * arrow_len)
                cv2.arrowedLine(frame, (screen_cx, screen_cy), (end_x, end_y), (255, 0, 255), 8)
            else:
                # ---------------------------
                # 🔴 场景 B: 丢失目标 (看不见码了!)
                # --------------------------
                # 判断是“高空丢失”还是“低空丢失”
                if last_valid_height < 0.5:
                    # === 【核心】低空盲降逻辑 ===
                    print(f"📉 进入盲降模式 (最后高度: {last_valid_height:.2f}m)")
                    # 1. 水平方向：绝对不动 (相信之前的对准)
                    cmd_r = 0.0
                    cmd_p = 0.0
                    # 2. 垂直方向：给一个能够触地的速度
                    vel_z = -0.15

                    # (可选) 超时保护：如果盲降了 3秒 还没停桨(也没触地)，就悬停报警
                    if blind_land_start_time == 0:
                        blind_land_start_time = time.time()
                    elif time.time() - blind_land_start_time > 3.0:
                        print("❌ 盲降超时！悬停！")
                        vel_z = 0.0
                    #可能存在识别不到但是位置很高的情况，

            # --- D. 异步发送 ---
            send_data = {
                "r": float(cmd_pitch),
                "p": float(cmd_roll),
                "y": 0.0,
                "t": float(vel_z)
            }
            udp_sender.send_async(send_data)

            # --- E. 显示 ---
            # 缩小一点显示，防止占满屏幕
            show_frame = cv2.resize(frame, (960, 540))
            # 显示 FPS
            fps = 1.0 / max(dt, 0.001)
            cv2.putText(show_frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("M3E Fusion Control", show_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("🛑 正在停止线程...")
        stream_loader.stop()
        udp_sender.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()