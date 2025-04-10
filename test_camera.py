import cv2

for index in range(5):
    for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG, 0]:
        print(f"尝试打开摄像头 index={index}, API={api}...")
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            print(f"✅ 成功打开摄像头 index={index}, API={api}")
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Camera Test", frame)
                cv2.waitKey(0)
            cap.release()
            cv2.destroyAllWindows()
            break
        cap.release()
