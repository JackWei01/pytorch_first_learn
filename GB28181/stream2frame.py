import cv2

# IP摄像头的URL（请替换成你的摄像头URL）
camera_url = "rtmp://192.168.222.129/live/test"
# camera_url = "http://your_camera_ip_address/video"

# 打开IP摄像头视频流Q
cap = cv2.VideoCapture(camera_url)

# 检查视频流是否成功打开
if not cap.isOpened():
    print("无法打开IP摄像头视频流")
    exit()

# 无限循环，读取视频流的每一帧
while True:
    # 读取一帧
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        print("无法读取帧")
        break

    # 在这里你可以对帧进行处理，或者保存帧成图像文件
    # 例如，保存每一帧成图片文件
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    image_filename = f"frame_{int(frame_number):04d}.jpg"
    cv2.imwrite(image_filename, frame)

    # 显示当前帧
    cv2.imshow("IP Camera Frame", frame)

    # 检查是否按下了 'q' 键，如果是则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频流资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
