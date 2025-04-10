import cv2
import numpy as np
import os

# ========== 参数 ==========
chessboard_size = (9, 6)     # 9x6 内角点（10x7 格子）
square_size = 0.025          # 单位：米（棋盘格单格边长，建议25mm）
save_file = "calibration_data.npz"
image_folder = "calib_manual"

# ========== 创建文件夹 ==========
os.makedirs(image_folder, exist_ok=True)

# ========== 相机初始化 ==========
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("📸 按空格拍照（建议拍 20 张），按 q 或 Esc 开始标定。")

img_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    display = frame.copy()
    if ret_cb:
        cv2.drawChessboardCorners(display, chessboard_size, corners, ret_cb)

    cv2.putText(display, f"已拍摄: {img_counter}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Calibration Capture", display)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
    elif key == 32:  # 空格保存
        img_path = os.path.join(image_folder, f"calib_{img_counter:02d}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"✅ 已保存：{img_path}")
        img_counter += 1

cap.release()
cv2.destroyAllWindows()

# ========== 读取图像进行标定 ==========
objpoints = []
imgpoints = []

objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # 实际尺寸

images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
print(f"\n📂 共读取 {len(images)} 张图像进行标定...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        objpoints.append(objp)

# ========== 相机标定 ==========
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n✅ 标定完成")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs.ravel())

np.savez(save_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print(f"📁 标定结果已保存为 {save_file}")
