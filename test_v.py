import socket
import json
import time

# ================= 配置区域 =================
# Android 手机 IP (请务必确认和手机在同一网段)
# USB 连接通常是 192.168.42.129，WIFI 连接请看手机设置
ANDROID_IP = "192.168.42.129"
ANDROID_PORT = 8888

# 测试速度 (米/秒)
# 0.25 是 M3E 的安全测试速度 (既能突破死区，又给人反应时间)
TEST_SPEED = 0.25


# ================= 主程序 =================
def main():
    # 1. 创建 UDP 套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"📡 UDP 就绪 -> 目标: {ANDROID_IP}:{ANDROID_PORT}")
    print(f"⚠️ 警告: 程序启动 3秒 后将强制发送 [向前 {TEST_SPEED} m/s] 的指令！")
    print("👉 请确保前方 5米 空旷，且手放在遥控器【切档开关】上随时准备切回 N 档刹车！")

    # 2. 倒计时 (给你一点心理准备时间)
    for i in range(3, 0, -1):
        print(f"⏳ {i}...")
        time.sleep(1)
    print("🚀 开始发送指令！按 Ctrl+C 停止")

    # 3. 构造固定的指令包
    # r=Roll(左右), p=Pitch(前后), y=Yaw(旋转), t=Throttle(上下)
    # p = +0.25 代表向前飞 (Body系)
    move_cmd = {
        "r": 0.0,
        "p": float(TEST_SPEED),
        "y": 0.0,
        "t": 0.0
    }

    # 停止指令 (用于退出时刹车)
    stop_cmd = {
        "r": 0.0,
        "p": 0.0,
        "y": 0.0,
        "t": 0.0
    }

    try:
        # 4. 死循环发送 (模拟虚拟摇杆的连续信号)
        while True:
            # 打包并发送
            msg = json.dumps(move_cmd).encode('utf-8')
            udp_socket.sendto(msg, (ANDROID_IP, ANDROID_PORT))

            print(f"📤 发送中: {move_cmd}")

            # 保持 30Hz 的频率 (0.033秒一次)
            # 频率太低飞机会自动刹车，太高会堵塞网络
            time.sleep(0.033)

    except KeyboardInterrupt:
        print("\n🛑 用户手动停止！正在发送刹车指令...")

        # 5. 安全退出：连续发送 10 次停止指令，确保飞机收到
        for _ in range(10):
            udp_socket.sendto(json.dumps(stop_cmd).encode('utf-8'), (ANDROID_IP, ANDROID_PORT))
            time.sleep(0.01)

        print("✅ 已发送停止指令，程序退出。")

    finally:
        udp_socket.close()


if __name__ == "__main__":
    main()