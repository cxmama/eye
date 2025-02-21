import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io
import pandas as pd

# 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 页面标题
st.title("瞳距测量（单位：毫米）")

# 侧边栏选择：上传图片或使用摄像头
option = st.sidebar.selectbox("选择输入方式", ["上传图片", "使用摄像头"])

# 侧边栏添加参考脸宽输入
REFERENCE_FACE_WIDTH_MM = st.sidebar.number_input("输入参考脸宽（毫米）", min_value=100.0, max_value=200.0, value=140.0, step=1.0)

def calculate_distance(point1, point2):
    """计算两点之间的欧几里得距离"""
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def measure_pupillary_distance(frame, results):
    """使用 MediaPipe 关键点测量瞳距"""
    if not results.multi_face_landmarks:
        return None, None, None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    
    # MediaPipe 关键点：左眼瞳孔 (468), 右眼瞳孔 (473)
    left_pupil = (int(landmarks[468].x * w), int(landmarks[468].y * h))
    right_pupil = (int(landmarks[473].x * w), int(landmarks[473].y * h))
    
    # 计算瞳距（像素）
    pd_px = calculate_distance(left_pupil, right_pupil)
    
    # 使用脸宽作为参考（关键点 33 和 263 近似脸颊两侧）
    left_cheek = (int(landmarks[33].x * w), int(landmarks[33].y * h))
    right_cheek = (int(landmarks[263].x * w), int(landmarks[263].y * h))
    face_width_px = calculate_distance(left_cheek, right_cheek)
    
    # 计算像素到毫米的比例
    pixel_to_mm_ratio = REFERENCE_FACE_WIDTH_MM / face_width_px
    
    # 转换为毫米
    pd_mm = round(pd_px * pixel_to_mm_ratio, 2)
    
    return {"瞳距 (PD)": pd_mm}, pixel_to_mm_ratio, (left_pupil, right_pupil)

def draw_measurements_on_lines(frame, pupil_points, pd_mm):
    """在图像上绘制瞳距测量结果"""
    left_pupil, right_pupil = pupil_points
    pd_mid_x = int((left_pupil[0] + right_pupil[0]) / 2)
    pd_mid_y = int((left_pupil[1] + right_pupil[1]) / 2)
    
    # 绘制瞳孔标记点
    cv2.circle(frame, left_pupil, 3, (0, 255, 0), -1)
    cv2.circle(frame, right_pupil, 3, (0, 255, 0), -1)
    
    # 绘制连接线和测量值
    cv2.line(frame, left_pupil, right_pupil, (0, 0, 255), 2)
    cv2.putText(frame, f"{pd_mm['瞳距 (PD)']} mm", (pd_mid_x - 20, pd_mid_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame

def get_csv_download(pd_mm):
    """生成包含瞳距的 CSV 文件"""
    df = pd.DataFrame([pd_mm], columns=["瞳距 (PD)"])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode('utf-8')

if option == "上传图片":
    uploaded_file = st.file_uploader("上传面部照片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # 读取上传的图片
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 处理图像
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            pd_mm, pixel_to_mm_ratio, pupil_points = measure_pupillary_distance(frame, results)
            if pd_mm:
                frame = draw_measurements_on_lines(frame, pupil_points, pd_mm)
                
                # 显示结果
                st.image(frame, caption="瞳距测量结果（毫米）", channels="BGR")
                st.write(f"像素到毫米比例: {pixel_to_mm_ratio:.4f} mm/px")

                # 提供 CSV 下载按钮
                csv_data = get_csv_download(pd_mm)
                st.download_button(
                    label="下载瞳距数据 (CSV, 毫米)",
                    data=csv_data,
                    file_name="pupillary_distance_mm.csv",
                    mime="text/csv",
                    key="download_csv_image"
                )
        else:
            st.write("未检测到面部。请上传清晰的面部照片。")

elif option == "使用摄像头":
    st.write("点击下方按钮开始摄像头测量")
    
    # 初始化 session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'pd_mm' not in st.session_state:
        st.session_state.pd_mm = None
    
    # 按钮控制
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("开始测量", key="start_webcam")
    with col2:
        stop_button = st.button("停止测量", key="stop_webcam")

    if start_button:
        st.session_state.running = True
    
    if stop_button:
        st.session_state.running = False

    # 摄像头处理
    if st.session_state.running:
        cap = cv2.VideoCapture(-1)
        if not cap.isOpened():
            st.error("无法打开摄像头。请检查设备是否连接或权限是否正确。")
            st.session_state.running = False
        else:
            stframe = st.empty()
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("无法读取摄像头画面。请检查设备。")
                    break

                # 处理帧
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    pd_mm, pixel_to_mm_ratio, pupil_points = measure_pupillary_distance(frame, results)
                    if pd_mm:
                        st.session_state.pd_mm = pd_mm
                        frame = draw_measurements_on_lines(frame, pupil_points, pd_mm)
                        stframe.image(frame, channels="BGR", caption="实时瞳距测量（毫米）")

                # 检查停止按钮（非实时响应，但避免阻塞）
                if stop_button:
                    st.session_state.running = False
                    break

            cap.release()
            cv2.destroyAllWindows()

    # 显示下载按钮
    if st.session_state.pd_mm:
        csv_data = get_csv_download(st.session_state.pd_mm)
        st.download_button(
            label="下载瞳距数据 (CSV, 毫米)",
            data=csv_data,
            file_name="pupillary_distance_mm.csv",
            mime="text/csv",
            key="download_csv_webcam"
        )

# 侧边栏说明
st.sidebar.write("古主任，这是试验版本.")
st.sidebar.write("古主任，仅用于测试")
