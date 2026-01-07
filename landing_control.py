import cv2
import cv2.aruco as aruco
import numpy as np
import time


# ================= 1. PID 控制器类 =================
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
        if dt <= 0: dt = 0.05

        # P
        p_term = self.kp * error

        # I (带限幅，防止积分饱和)
        self.integral += error * dt
        # 简单的积分抗饱和
        if self.integral > 1.0:
            self.integral = 1.0
        elif self.integral < -1.0:
            self.integral = -1.0
        i_term = self.ki * self.integral

        # D
        d_term = self.kd * (error - self.prev_error) / dt

        output = p_term + i_term + d_term

        self.prev_error = error
        self.last_time = current_time

        # 输出限幅
        return max(min(output, self.max_out), -self.max_out)


# ================= 2. 配置区域 =================
# 真实物理尺寸 (米)
TAG_SIZE_BIG = 0.60
TAG_SIZE_SMALL = 0.12

# 【关键】偏移量配置
# 格式: ID: np.array([x_offset, y_offset, z_offset])
# 含义: 该 Tag 中心相对于【整个降落板中心】的位置
# 请根据你的实际打印纸测量修改这些值！
# 下面是假设值：假设小码中心距离大中心 15cm
TAG_LAYOUT = {
    0: np.array([0.0, 0.0, 0.0]),  # 大码本身就是中心
    576: np.array([0.15, -0.15, 0.0]),  # 右上 (图像坐标系: 右X+, 下Y+)
    571: np.array([-0.15, 0.15, 0.0])  # 左下
}

# PID 参数 (需要实飞调试，这是经验值)
# P: 响应速度, D: 阻尼(防止震荡), I: 消除静差
pid_x = PIDController(kp=1.5, ki=0.0, kd=0.5, max_out=1.0)  # 控制左右
pid_y = PIDController(kp=1.5, ki=0.0, kd=0.5, max_out=1.0)  # 控制前后
pid_z = PIDController(kp=0.8, ki=0.0, kd=0.1, max_out=0.5)  # 控制下降
pid_yaw = PIDController(kp=1.0, ki=0.0, kd=0.1, max_out=30)  # 控制旋转


def main():
    # 模拟相机内参 (换成你之前的 get_dji_camera_matrix)
    # 这里为了演示直接写死，实际请用之前的函数计算
    f_val = 1223.3
    camera_matrix = np.array([[f_val, 0, 960], [0, f_val, 540], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))

    # 准备检测
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # 读取图片模拟视频流
    frame = cv2.imread("tag21.png")
    if frame is None: return
    # 强制 Resize 模拟真实情况
    frame = cv2.resize(frame, (1420, 1380))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # 用于存储所有检测到的 Tag 算出来的“板子中心位置”
        pad_positions = []

        for i in range(len(ids)):
            curr_id = ids[i][0]

            # 1. 确定当前用多大的尺寸算
            if curr_id in [571, 576]:
                size = TAG_SIZE_SMALL
            else:
                size = TAG_SIZE_BIG

            # 2. PnP 解算
            obj_points = np.array([[-size / 2, size / 2, 0], [size / 2, size / 2, 0], [size / 2, -size / 2, 0],
                                   [-size / 2, -size / 2, 0]], dtype=np.float32)
            ret, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)

            # tvec 是: 相机 看 Tag 的位置
            # 我们需要: 相机 看 【板子中心】 的位置
            # 公式: Pos_Center = Pos_Tag - Offset
            # 注意：这里是简化计算，假设平面平行，直接减去偏移量
            # 如果 Tag 有旋转，需要用旋转矩阵变换 offset，但降落时通常忽略这个微小差异

            if curr_id in TAG_LAYOUT:
                offset = TAG_LAYOUT[curr_id]
                # 修正后的坐标
                corrected_pos = tvec.flatten() - offset
                pad_positions.append(corrected_pos)

                print(f"ID {curr_id} 原生X: {tvec[0][0]:.2f} -> 修正后X: {corrected_pos[0]:.2f}")

        # 3. 数据融合 (取平均值)
        if pad_positions:
            avg_pos = np.mean(pad_positions, axis=0)
            final_x, final_y, final_z = avg_pos

            print(f"\n======== ✈️ 最终决策数据 ========")
            print(f"高度: {final_z:.2f}m")
            print(f"偏差: X={final_x:.2f}m, Y={final_y:.2f}m")

            # 4. PID 计算控制量
            # 目标是 X=0, Y=0 (图像中心)
            # 飞机的 roll 控制左右 (对应图像 X)
            # 飞机的 pitch 控制前后 (对应图像 Y)

            # 注意方向：
            # 如果图像 X > 0 (板子在画面右边)，飞机需要往右飞 (Roll > 0)
            # 如果图像 Y > 0 (板子在画面下边)，飞机需要往后飞 (Pitch < 0) -> 这里要反相！或者看你飞机的定义

            # 假设机头朝前，相机朝下，画面上方是机头方向：
            # 画面 X+ (右) -> 飞机 Roll+ (右移)
            # 画面 Y+ (下) -> 飞机 Pitch- (后退) !!!

            vel_x = pid_x.update(0 - final_x)  # 目标 0 - 当前 x
            vel_y = pid_y.update(0 - final_y)

            # 高度策略：如果水平对其了，就开始下降
            if abs(final_x) < 0.2 and abs(final_y) < 0.2:
                vel_z = -0.5  # 下降速度 0.5 m/s
            else:
                vel_z = 0.0  # 先不降，对准再说

            # 这里的 vel_x (Roll速度), vel_y (Pitch速度) 就是要发给 MSDK 的
            # 注意: 这里输出的是 PID 原始值，方向可能需要根据实飞取反
            print(f"发送指令 -> Roll(Vx): {-vel_x:.2f}, Pitch(Vy): {vel_y:.2f}, Vertical: {vel_z}")

            # 可视化中心点
            img_center_x = int(1920 / 2 + final_x * (1223 / final_z))  # 简单投影画一下
            img_center_y = int(1080 / 2 + final_y * (1223 / final_z))
            cv2.circle(frame, (img_center_x, img_center_y), 20, (0, 255, 255), -1)
            cv2.putText(frame, "TARGET", (img_center_x + 30, img_center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                        2)

            cv2.imshow("PID Landing", cv2.resize(frame, (960, 540)))
            cv2.waitKey(0)
if __name__ == '__main__':
    main()