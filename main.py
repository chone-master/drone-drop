# import cv2
# import cv2.aruco as aruco
# import sys
# import os
#
#
# def main():
#     # ================= 配置区域 =================
#     # 图片文件名
#     image_path = "tag6.png"
#     # ===========================================
#
#     if not os.path.exists(image_path):
#         print(f"❌ 错误：找不到文件 '{image_path}'")
#         return
#
#     frame = cv2.imread(image_path)
#     if frame is None:
#         print("❌ 错误：无法读取图片")
#         return
#
#     print(f"✅ 成功读取图片: {image_path} ({frame.shape[1]}x{frame.shape[0]})")
#
#     # 准备要测试的字典
#     test_dicts = {
#         "AprilTag 36h11": aruco.DICT_APRILTAG_36h11,
#         "ArUco 6x6_250": aruco.DICT_6X6_250,
#         "ArUco 5x5_250": aruco.DICT_5X5_250,
#         "ArUco 4x4_250": aruco.DICT_4X4_250
#     }
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     found_any = False
#
#     print("\n🔍 开始检测...")
#
#     for name, dict_enum in test_dicts.items():
#         aruco_dict = aruco.getPredefinedDictionary(dict_enum)
#         parameters = aruco.DetectorParameters()
#
#         # =========== 核心修改在这里 ===========
#         # 新版 OpenCV (4.7+) 写法：
#         # 1. 先创建检测器对象
#         detector = aruco.ArucoDetector(aruco_dict, parameters)
#
#         # 2. 使用检测器对象进行检测 (注意参数变少了，不用传 dict 和 parameters 了)
#         corners, ids, rejected = detector.detectMarkers(gray)
#         # =====================================
#
#         if ids is not None:
#             found_any = True
#             count = len(ids)
#             print(f"✨ 命中字典 [{name}] -> 检测到 {count} 个 Tag！IDs: {ids.flatten()}")
#
#             # 绘制结果
#             aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 255, 0))
#
#             # 标记文字
#             cv2.putText(frame, f"Dict: {name}", (20, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
#             cv2.putText(frame, f"IDs: {ids.flatten()}", (20, 100),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#             break
#
#     if not found_any:
#         print("❌ 未检测到任何 Tag。")
#     else:
#         # 显示结果
#         scale_percent = 50
#         if frame.shape[1] > 1920:
#             width = int(frame.shape[1] * scale_percent / 100)
#             height = int(frame.shape[0] * scale_percent / 100)
#             dim = (width, height)
#             frame_show = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#         else:
#             frame_show = frame
#
#         cv2.imshow("Result", frame_show)
#         print("\n按任意键关闭窗口...")
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()


import cv2
import cv2.aruco as aruco
import sys
import os


def main():
    # ================= 配置区域 =================
    # 这里改成你的 mp4 文件名
    video_path = "test1.mp4"
    # ===========================================

    # 1. 使用 VideoCapture 读取视频
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频文件 '{video_path}'")
        return

    print(f"✅ 成功打开视频: {video_path}")
    print("按 'q' 键或 'ESC' 键退出程序")

    # 准备要测试的字典
    test_dicts = {
        "AprilTag 36h11": aruco.DICT_APRILTAG_36h11,
        "ArUco 6x6_250": aruco.DICT_6X6_250,
        "ArUco 5x5_250": aruco.DICT_5X5_250,
        "ArUco 4x4_250": aruco.DICT_4X4_250
    }

    # 2. 进入视频循环处理每一帧
    while True:
        ret, frame = cap.read()

        # 如果读不到帧（视频结束或出错），则退出循环
        if not ret:
            print("视频播放结束或无法读取帧。")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found_any_in_frame = False

        # 遍历字典进行检测
        for name, dict_enum in test_dicts.items():
            aruco_dict = aruco.getPredefinedDictionary(dict_enum)
            parameters = aruco.DetectorParameters()

            # =========== 核心检测逻辑 (OpenCV 4.7+) ===========
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejected = detector.detectMarkers(gray)
            # ================================================

            if ids is not None:
                found_any_in_frame = True

                # 绘制结果
                aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 255, 0))

                # 标记文字
                cv2.putText(frame, f"Dict: {name}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # 这里为了防止ID太多遮挡屏幕，只显示前几个ID
                ids_str = str(ids.flatten()) if len(ids) < 5 else str(ids.flatten()[:5]) + "..."
                cv2.putText(frame, f"IDs: {ids_str}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # 如果在这个字典找到了，就跳出字典循环，不再尝试其他字典（避免闪烁）
                break

        # 显示处理后的画面
        # 如果分辨率太高（比如4K视频），缩小显示
        if frame.shape[1] > 1920:
            scale_percent = 50
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame_show = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        else:
            frame_show = frame

        cv2.imshow("Video Result", frame_show)

        # 3. 键盘控制退出
        # waitKey(1) 表示等待1毫秒，这会让视频连续播放
        # 如果用 waitKey(0) 则会暂停在每一帧
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 按 'q' 或 ESC 退出
            break

    # 4. 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()