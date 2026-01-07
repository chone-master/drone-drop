import cv2
import cv2.aruco as aruco
import numpy as np

# ================= 配置区域 =================
TAG_SIZE_BIG = 0.60
TAG_SIZE_SMALL = 0.12

# 偏移量配置 (请根据你的实际贴纸测量调整)
TAG_LAYOUT = {
    0: np.array([0.0, 0.0, 0.0]),
    576: np.array([0.15, -0.15, 0.0]),  # 假设：右上
    571: np.array([-0.15, 0.15, 0.0])  # 假设：左下
}


def try_force_detect_big_tag(detector, frame, current_ids):
    """
    暴力尝试检测大码 (ID 0)
    策略：如果当前没找到ID 0，就对图像进行【膨胀处理】，抹除小码的细节
    """
    # 如果已经找到了 ID 0，就直接返回空，不用费劲了
    if current_ids is not None and 0 in current_ids:
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === 核心魔法：膨胀处理 ===
    # 计算一个核大小，大约是图像宽度的 2% (足以抹掉小码，但保留大码)
    k_size = int(frame.shape[1] * 0.02)
    if k_size % 2 == 0: k_size += 1  # 必须是奇数

    kernel = np.ones((k_size, k_size), np.uint8)

    # 膨胀：让白色区域扩张，吃掉小码的黑色纹理
    dilated_img = cv2.dilate(gray, kernel, iterations=1)

    # 再次检测
    corners, ids, _ = detector.detectMarkers(dilated_img)

    if ids is not None:
        for i in range(len(ids)):
            if ids[i][0] == 0:  # 只有当它是我们要找的大码时
                print(f"💡 [魔法生效] 通过图像膨胀找回了 ID: 0")
                # 可选：显示一下处理后的图看看效果
                # cv2.imshow("Dilated View (For Big Tag)", cv2.resize(dilated_img, (480, 480)))
                return [corners[i]], np.array([[0]])

    return None, None


def main():
    # 模拟相机内参
    f_val = 1223.3
    camera_matrix = np.array([[f_val, 0, 960], [0, f_val, 540], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # 读取图片
    frame = cv2.imread("tag.png")  # 你的文件名
    if frame is None: return
    # 强制 resize 到 1080P 模拟推流
    frame = cv2.resize(frame, (340, 340))

    # 1. 第一轮：正常检测 (找小码)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # 2. 第二轮：如果没有 ID 0，尝试“魔法检测”
    corners = list(corners) if corners else []
    ids = list(ids) if ids is not None else []

    # 检查现有 ID 里有没有 0
    has_id_0 = False
    for id_arr in ids:
        if id_arr[0] == 0: has_id_0 = True

    if not has_id_0:
        # 尝试暴力找大码
        new_corner, new_id = try_force_detect_big_tag(detector, frame, np.array(ids))
        if new_corner:
            corners.extend(new_corner)
            ids.append(new_id[0])

    ids = np.array(ids)  # 转回 numpy 方便后续处理

    # === 下面是数据融合逻辑 ===
    if len(ids) > 0:
        print(f"✅ 最终检测列表: {ids.flatten()}")
        pad_positions = []

        for i in range(len(ids)):
            curr_id = ids[i][0]
            curr_corners = corners[i]

            # 切换尺寸
            if curr_id in [571, 576]:
                size = TAG_SIZE_SMALL
                tag_type = "小码"
            else:
                size = TAG_SIZE_BIG
                tag_type = "大码"

            # PnP 解算
            obj_points = np.array([[-size / 2, size / 2, 0], [size / 2, size / 2, 0], [size / 2, -size / 2, 0],
                                   [-size / 2, -size / 2, 0]], dtype=np.float32)
            ret, rvec, tvec = cv2.solvePnP(obj_points, curr_corners, camera_matrix, dist_coeffs)

            # 坐标系修正
            if curr_id in TAG_LAYOUT:
                offset = TAG_LAYOUT[curr_id]
                corrected_pos = tvec.flatten() - offset
                pad_positions.append(corrected_pos)

                print(f"Target: {tag_type} (ID {curr_id})")
                print(f"  > 修正后坐标: X={corrected_pos[0]:.2f}, Y={corrected_pos[1]:.2f}, Z={tvec[2][0]:.2f}")

        # 融合结果
        if pad_positions:
            avg_pos = np.mean(pad_positions, axis=0)
            print(f"\n======== ✈️ 融合成功 ========")
            print(f"高度: {avg_pos[2]:.2f}m")
            print(f"偏差: X={avg_pos[0]:.2f}m, Y={avg_pos[1]:.2f}m")

            # 可视化
            vis_corners = np.array([c for c in corners])
            vis_ids = np.array(ids)
            aruco.drawDetectedMarkers(frame, vis_corners, vis_ids)
            cv2.imshow("Final Result", cv2.resize(frame, (960, 540)))
            cv2.waitKey(0)
    else:
        print("❌ 依然未检测到 Tag，请检查光照或图片清晰度")


if __name__ == "__main__":
    main()