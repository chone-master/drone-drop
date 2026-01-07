import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time
import socket  # 【新增】引入网络通信库
import json  # 【新增】引入JSON库

# ================= 1. 用户配置区域 =================
VIDEO_PATH = "test2.mp4"  # 你的视频文件路径
# VIDEO_PATH = 0          # 如果设为 0，则调用摄像头

# 【新增】网络配置
# 请务必修改为你 Android 手机的实际 IP 地址
# 如果使用 USB 网络共享，通常是 192.168.42.129
ANDROID_IP = "192.168.42.129"
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


# ================= 2. 工具类 =================

class PIDController:
    def __init__(self, kp, ki, kd, max_out=1.0):
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


# ================= 3. 主程序 =================

def main():
    # 1. 初始化视觉参数
    K, D = get_camera_matrix(STREAM_W, STREAM_H, DFOV_DEG)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, params)

    # 2. 初始化 PID
    pid_roll = PIDController(kp=1.0, ki=0.0, kd=0.1)  # 控制左右
    pid_pitch = PIDController(kp=1.0, ki=0.0, kd=0.1)  # 控制前后

    # 3. 【新增】初始化 UDP Socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"📡 UDP 通信已就绪，目标地址: {ANDROID_IP}:{ANDROID_PORT}")

    # 4. 打开视频源
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {VIDEO_PATH}")
        return

    print(f"🎬 开始播放视频测试... (按 'q' 退出, 'SPACE' 暂停)")
    is_paused = False

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("✅ 视频播放结束")
                break

            if frame.shape[1] != STREAM_W:
                frame = cv2.resize(frame, (STREAM_W, STREAM_H))

            # --- 核心处理逻辑 ---
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

            # --- 决策与发送 ---
            if detected_positions:
                final_pos = np.mean(detected_positions, axis=0)
                x, y, z = final_pos

                # 计算 PID 指令
                cmd_roll = pid_roll.update(0 - x, dt=0.033) * -1  # 左右速度
                cmd_pitch = pid_pitch.update(0 - y, dt=0.033)  # 前后速度

                # 【新增】简单的垂直速度控制策略
                # 逻辑：只有当水平误差小于 0.2 米时，才允许缓慢下降
                horizontal_error = math.sqrt(x ** 2 + y ** 2)
                vel_z = 0.0
                if horizontal_error < 0.2:
                    vel_z = -0.3  # 下降速度 -0.3 m/s

                # 【新增】打包并发送 UDP 数据
                # 数据格式必须与 Android 端解析逻辑一致: {"r":.., "p":.., "y":.., "t":..}
                send_data = {
                    "r": float(cmd_roll),
                    "p": float(cmd_pitch),
                    "y": 0.0,  # 暂不控制旋转
                    "t": float(vel_z)
                }

                try:
                    message = json.dumps(send_data).encode('utf-8')
                    udp_socket.sendto(message, (ANDROID_IP, ANDROID_PORT))
                    # print(f"📡 发送: {send_data}") # 调试时可以取消注释
                except Exception as e:
                    print(f"❌ UDP 发送失败: {e}")

                # --- 绘制 UI ---
                cx = int(STREAM_W / 2 + x * (K[0][0] / z))
                cy = int(STREAM_H / 2 + y * (K[1][1] / z))
                screen_cx, screen_cy = STREAM_W // 2, STREAM_H // 2

                cv2.line(frame, (screen_cx, screen_cy), (cx, cy), (0, 255, 0), 3)
                cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)

                info_text = [
                    f"Height: {z:.2f}m",
                    f"Err X: {x:.2f}m",
                    f"Err Y: {y:.2f}m",
                    f"CMD R: {cmd_roll:.2f}",
                    f"CMD P: {cmd_pitch:.2f}",
                    f"CMD Z: {vel_z:.2f}"  # 把 Z 轴速度也显示出来
                ]

                for idx, text in enumerate(info_text):
                    cv2.putText(frame, text, (50, 100 + idx * 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)

                arrow_len = 150
                end_x = int(screen_cx + cmd_roll * arrow_len)
                end_y = int(screen_cy - cmd_pitch * arrow_len)
                cv2.arrowedLine(frame, (screen_cx, screen_cy), (end_x, end_y), (255, 0, 255), 8)

            else:
                cv2.putText(frame, "SEARCHING...", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)

                # 【新增】丢失目标时的保护逻辑
                # 当看不到码时，发送全 0 指令让飞机悬停，防止它乱飘
                try:
                    stop_data = {"r": 0.0, "p": 0.0, "y": 0.0, "t": 0.0}
                    udp_socket.sendto(json.dumps(stop_data).encode('utf-8'), (ANDROID_IP, ANDROID_PORT))
                except:
                    pass

            show_frame = cv2.resize(frame, (960, 540))
            cv2.imshow("Video Playback Test", show_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_paused = not is_paused

    # 清理资源
    cap.release()
    udp_socket.close()  # 【新增】关闭Socket
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()