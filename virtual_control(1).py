import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time
import socket
import json
import threading
import queue

# ================= 1. 用户配置区域 =================
# ====== RTSP 配置 ======
RTSP_USERNAME = "djiuser"
RTSP_PASSWORD = "123456"
RTSP_IP = "192.168.31.214"  # 遥控器 IP
RTSP_PORT = 8554
RTSP_PATH = "streaming/live/1"

# 【新增】网络配置
ANDROID_IP = "172.27.46.80"
ANDROID_PORT = 8888

# 分辨率与相机参数
STREAM_W = 3840
STREAM_H = 2160
DFOV_DEG = 84.0

# 物理尺寸 (米)
TAG_SIZE_BIG = 0.515
TAG_SIZE_SMALL = 0.096

# 布局偏移量
TAG_LAYOUT = {
    0: np.array([0.0, 0.0, 0.0]),
    576: np.array([0.15, -0.15, 0.0]),
    571: np.array([-0.15, 0.15, 0.0])
}


# ================= 2. 多线程工具类 (核心修改) =================

class RTSPStreamLoader:
    """
    线程1：RTSP 拉流线程
    作用：不断从网络读取最新帧，确保主程序处理的永远是最新画面，
    解决 cv2.read() 阻塞导致的延迟积压问题。
    """

    def __init__(self, rtsp_url):
        self.connect_success = False
        self.frame = None
        self.stopped = False
        self.rtsp_url = rtsp_url
        self.lock = threading.Lock()

        print(f"📡 正在连接 RTSP 流: {rtsp_url} ...")
        self.cap = cv2.VideoCapture(rtsp_url)
        if self.cap.isOpened():
            self.connect_success = True
            print("✅ RTSP 连接成功！")
        else:
            print("❌ RTSP 连接失败！")

    def start(self):
        if self.connect_success:
            # 开启守护线程
            t = threading.Thread(target=self.update, args=(), daemon=True)
            t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                # 断线重连逻辑
                print("⚠️ RTSP 断开，尝试重连...")
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.rtsp_url)
                continue

            ret, frame = self.cap.read()
            if ret:
                # 使用锁确保写操作安全，只保留最新的一帧
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)  # 防止死循环占用CPU

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()


class UDPSender:
    """
    线程2：UDP 发送线程
    作用：异步发送数据，防止网络波动阻塞主程序的图像处理
    """

    def __init__(self, target_ip, target_port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_addr = (target_ip, target_port)
        self.queue = queue.Queue(maxsize=10)  # 队列，防止积压
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.run, args=(), daemon=True)
        t.start()
        return self

    def send_async(self, data_dict):
        # 如果队列满了，说明发送太慢，丢弃旧指令，保证实时性
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put(data_dict)

    def run(self):
        while not self.stopped:
            try:
                # 阻塞等待新指令，超时1秒
                data = self.queue.get(timeout=1)
                json_str = json.dumps(data).encode('utf-8')
                self.sock.sendto(json_str, self.target_addr)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ UDP 发送错误: {e}")

    def stop(self):
        self.stopped = True
        self.sock.close()


# ================= 3. 算法工具类 =================
def apply_deadband(val):
    MIN_MOVE_SPEED = 0.12
    STOP_THRESHOLD = 0.05
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


def get_camera_matrix(width, height, dfov_deg):
    diagonal_pixels = math.sqrt(width ** 2 + height ** 2)
    fov_rad = math.radians(dfov_deg)
    f_px = (diagonal_pixels / 2) / math.tan(fov_rad / 2)
    K = np.array([[f_px, 0, width / 2], [0, f_px, height / 2], [0, 0, 1]], dtype=np.float32)
    D = np.zeros((5, 1))
    return K, D


# ================= 4. 主程序 (主线程：负责计算与显示) =================
def main():
    # 1. 初始化参数
    K, D = get_camera_matrix(STREAM_W, STREAM_H, DFOV_DEG)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, params)

    pid_roll = PIDController(kp=1.0, ki=0.0, kd=0.1)
    pid_pitch = PIDController(kp=1.0, ki=0.0, kd=0.1)

    # 2. 启动 RTSP 线程
    rtsp_url = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}/{RTSP_PATH}"
    stream_loader = RTSPStreamLoader(rtsp_url).start()

    # 3. 启动 UDP 线程
    udp_sender = UDPSender(ANDROID_IP, ANDROID_PORT).start()

    print("🚀 系统全速运行中... (按 'q' 退出)")

    # 等待第一帧
    while stream_loader.get_frame() is None:
        time.sleep(0.1)

    last_time = time.time()

    try:
        while True:
            # --- 步骤A: 从线程取最新帧 (非阻塞) ---
            frame = stream_loader.get_frame()
            if frame is None: continue

            # 计算真实的 dt (处理间隔)
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # 缩放处理 (节省算力)
            if frame.shape[1] != STREAM_W:
                frame = cv2.resize(frame, (STREAM_W, STREAM_H))

            # --- 步骤B: ArUco 检测 (计算密集型) ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            detected_positions = []
            if ids is not None:
                for i, id_val in enumerate(ids[:, 0]):
                    size = TAG_SIZE_SMALL if id_val in [571, 576] else TAG_SIZE_BIG
                    obj_pts = np.array([[-size / 2, size / 2, 0], [size / 2, size / 2, 0],
                                        [size / 2, -size / 2, 0], [-size / 2, -size / 2, 0]], dtype=np.float32)
                    _, rvec, tvec = cv2.solvePnP(obj_pts, corners[i], K, D)

                    pos_cam = tvec.flatten()
                    if id_val in TAG_LAYOUT:
                        pos_center = pos_cam - TAG_LAYOUT[id_val]
                        detected_positions.append(pos_center)
                        cv2.drawFrameAxes(frame, K, D, rvec, tvec, size)

            # --- 步骤C: PID 与 发送指令 ---
            cmd_roll, cmd_pitch, vel_z = 0.0, 0.0, 0.0  # 默认悬停

            if detected_positions:
                final_pos = np.mean(detected_positions, axis=0)
                x, y, z = final_pos

                raw_roll = pid_roll.update(0 - x, dt=dt) * -1
                raw_pitch = pid_pitch.update(0 - y, dt=dt)

                cmd_roll = apply_deadband(raw_roll)
                cmd_pitch = apply_deadband(raw_pitch)

                horizontal_error = math.sqrt(x ** 2 + y ** 2)
                vel_z = -0.3 if horizontal_error < 0.2 else 0.0

                # 绘制辅助线
                cx = int(STREAM_W / 2 + x * (K[0][0] / z))
                cy = int(STREAM_H / 2 + y * (K[1][1] / z))
                cv2.line(frame, (STREAM_W // 2, STREAM_H // 2), (cx, cy), (0, 255, 0), 3)
                cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)

            # 打包数据
            send_data = {"r": float(cmd_roll), "p": float(cmd_pitch), "y": 0.0, "t": float(vel_z)}

            # --- 步骤D: 扔给 UDP 线程发送 (非阻塞) ---
            udp_sender.send_async(send_data)

            # --- 步骤E: 界面显示 ---
            # 为了流畅，显示可以稍微降采样，或者直接显示
            show_frame = cv2.resize(frame, (960, 540))
            cv2.putText(show_frame, f"FPS: {1.0 / max(dt, 0.001):.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("M3E Vision Control", show_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 清理资源
        print("🛑 正在停止线程...")
        stream_loader.stop()
        udp_sender.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()