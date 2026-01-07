import cv2
import cv2.aruco as aruco
import numpy as np

# ================= 配置区域 =================
TAG_SIZE_BIG = 0.60
TAG_SIZE_SMALL = 0.12

TAG_LAYOUT = {
    0: np.array([0.0, 0.0, 0.0]),
    576: np.array([0.15, -0.15, 0.0]),
    571: np.array([-0.15, 0.15, 0.0])
}


def detect_with_surgical_mask(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. 原图检测
    corners1, ids1, _ = detector.detectMarkers(gray)

    final_corners = list(corners1) if corners1 else []
    final_ids_list = list(ids1[:, 0]) if ids1 is not None else []

    # 找出小码的位置
    small_tags_corners = []
    if ids1 is not None:
        for i, id_val in enumerate(ids1[:, 0]):
            if id_val in [571, 576]:
                small_tags_corners.append(corners1[i])

    # 2. 如果没找到大码，执行强力手术
    if 0 not in final_ids_list and len(small_tags_corners) > 0:
        print("💡 启动强力修复模式...")

        masked_gray = gray.copy()

        for corner_set in small_tags_corners:
            pts = corner_set[0]
            center = np.mean(pts, axis=0)

            # 【关键修改】扩大遮盖范围到 1.6 倍！宁可多盖，不可少盖
            expanded_pts = (pts - center) * 1.6 + center
            points_to_draw = expanded_pts.astype(np.int32)

            # 涂白
            cv2.fillPoly(masked_gray, [points_to_draw], 255)

        # 【关键修改】强制显示手术后的图，让你亲眼看看修得干不干净！
        # 如果你的 Orin 是无头模式（没接显示器），请注释掉下面这两行
        debug_view = cv2.resize(masked_gray, (960, 540))
        cv2.imshow("DEBUG: Masked Image", debug_view)
        cv2.waitKey(500)  # 停顿0.5秒让你看清楚

        # 再次检测
        corners2, ids2, _ = detector.detectMarkers(masked_gray)

        if ids2 is not None:
            for i, id_val in enumerate(ids2[:, 0]):
                if id_val == 0:
                    print("✨ [手术成功] 大码 ID: 0 复活了！")
                    final_corners.append(corners2[i])
                    final_ids_list.append(id_val)

    if not final_ids_list:
        return tuple(), None
    else:
        return tuple(final_corners), np.array(final_ids_list).reshape(-1, 1)


def main():
    # 模拟相机内参
    f_val = 1223.3
    camera_matrix = np.array([[f_val, 0, 960], [0, f_val, 540], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # 读取图片
    frame = cv2.imread("tag5.png")
    if frame is None: return
    frame = cv2.resize(frame, (1920, 1080))

    print(f"🔍 开始检测...")
    corners, ids = detect_with_surgical_mask(detector, frame)

    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten()
        print(f"✅ 当前列表: {ids_flat}")

        # 只要列表里有 0，我们就赢了
        if 0 in ids_flat:
            print("\n🎉🎉🎉 恭喜！大码小码全部识别成功！🎉🎉🎉")
        else:
            print("\n⚠️ 依然只有小码。请看弹出的 DEBUG 窗口，是不是白色框没盖住黑边？")

        # 这里的后续 PnP 代码省略，之前已经跑通了
        cv2.imshow("Result", cv2.resize(frame, (960, 540)))
        cv2.waitKey(0)
    else:
        print("❌ 未检测到任何 Tag")


if __name__ == "__main__":
    main()