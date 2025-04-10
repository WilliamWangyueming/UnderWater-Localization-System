import cv2
import numpy as np
from pupil_apriltags import Detector
from collections import deque
import time
import threading
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
import webbrowser
from flask import Flask, Response

# ====== 全局共享数据（线程安全） ======
data_lock = threading.Lock()
smoothed_pos = np.array([0.0, 0.0, 0.0])
robot_positions = []
update_enabled = True
frame_to_show = None
last_success_time = time.time()  # 新增：记录最后一次成功读取帧的时间

# ====== 相机参数加载 ======
calib_path = r'C:\Users\willi\src\calibration_data.npz'
with np.load(calib_path) as X:
    camera_matrix, dist_coeffs = X['camera_matrix'], X['dist_coeffs']

# ====== 摄像头初始化函数 ======
def create_video_capture():
    cap = cv2.VideoCapture("rtsp://192.168.166.78:8554/unicast")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap

cap = None  # 让 vision_thread() 自行创建

def safe_read(cap, timeout=1.5):
    """带超时的 cap.read()，避免 RTSP 卡死。"""
    result = [False, None]

    def reader():
        result[0], result[1] = cap.read()

    thread = threading.Thread(target=reader)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("[WARNING] cap.read() timeout.")
        return False, None
    return result[0], result[1]


# ====== AprilTag 检测器初始化 ======
at_detector = Detector(
    families='tag25h9',
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# ====== 姿态变换函数 ======
def get_transformation(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def draw_axes(image, corners, rvec, tvec):
    axis_len = 0.02
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, -axis_len],
    ])
    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    c = tuple(imgpts[0])
    pt_x = tuple(imgpts[1])
    pt_y = tuple(imgpts[2])
    pt_z = tuple(imgpts[3])

    image = cv2.line(image, c, pt_x, (0, 0, 255), 2)
    image = cv2.line(image, c, pt_y, (0, 255, 0), 2)
    image = cv2.line(image, c, pt_z, (255, 0, 0), 2)
    return image

# ====== 后台线程：处理摄像头、识别、坐标转换 ======
def vision_thread():
    global frame_to_show, cap, smoothed_pos, robot_positions, update_enabled, last_success_time

    tag_size = 0.038
    cam_pos_buffer = deque(maxlen=5)
    T_world_from_cam = None
    prev_time = time.time()
    failure_count = 0

    cap = None  # 初始为 None

    while True:
        if cap is None or not cap.isOpened():
            print("[INFO] Connecting to RTSP stream...")
            cap = create_video_capture()
            time.sleep(1.0)
            continue

        ret, frame = safe_read(cap, timeout=1.5)
        current_time = time.time()

        if ret:
            last_success_time = current_time
            failure_count = 0
        else:
            failure_count += 1

        # 状态更新：失败 3 次就变红（大约 0.1 秒 x 3）
        with data_lock:
            update_enabled = failure_count < 2

        # 重连条件：连续失败太多次（大约 1 秒以上）
        if failure_count > 30:
            print("[WARNING] Too many failures, reconnecting...")
            try:
                cap.release()
            except:
                pass
            cap = None
            time.sleep(2.0)
            continue

        if not ret:
            continue  # 跳过图像处理

        # ========== 图像处理和位姿估计 ==========
        curr_time = current_time
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = at_detector.detect(gray)
        tag_positions_world = {}
        robot_center_candidates = []

        marker_offsets = {
            1: np.array([-0.17, 0.10, 0.0]),
            2: np.array([-0.17, -0.10, 0.0]),
            3: np.array([0.17, -0.10, 0.0]),
            4: np.array([0.17, 0.10, 0.0]),
        }

        for det in detections:
            corners = det.corners.astype(np.float32)
            half = tag_size / 2
            objp = np.array([
                [-half, -half, 0],
                [half, -half, 0],
                [half, half, 0],
                [-half, half, 0],
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                continue

            tag_id = det.tag_id
            T_cam_from_tag = get_transformation(rvec, tvec)
            T_tag_from_cam = invert_transform(T_cam_from_tag)

            if tag_id == 0:
                T_world_from_cam = T_tag_from_cam.copy()

            if T_world_from_cam is not None:
                T_world_from_tag = T_world_from_cam @ T_cam_from_tag
                pos_world = T_world_from_tag[:3, 3]
                tag_positions_world[tag_id] = pos_world
                if tag_id in marker_offsets:
                    robot_center = pos_world - marker_offsets[tag_id]
                    robot_center_candidates.append(robot_center)

            frame = draw_axes(frame, corners, rvec, tvec)

        if T_world_from_cam is not None:
            cam_pos = T_world_from_cam[:3, 3]
            cam_pos_buffer.append(cam_pos)
            with data_lock:
                smoothed_pos = np.mean(cam_pos_buffer, axis=0)
                if robot_center_candidates:
                    avg_center = np.mean(robot_center_candidates, axis=0)
                    robot_positions = [(999, avg_center)]

        with data_lock:
            if not update_enabled:
                cv2.putText(frame, 'RTSP STREAM LOST', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            frame_to_show = frame.copy()


threading.Thread(target=vision_thread, daemon=True).start()



server = Flask(__name__)
app = dash.Dash(__name__, server=server)

@server.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with data_lock:
                global update_enabled  # 确保使用的是全局变量
                if frame_to_show is not None:
                    ret, jpeg = cv2.imencode('.jpg', frame_to_show, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(1/30)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div([
    html.H1("Underwater Perception System", style={
        'fontWeight': 'bold', 'fontSize': '48px', 'letterSpacing': '1px',
        'textTransform': 'uppercase','textAlign': 'center', 'margin': '0', 'padding': '10px', 'color': 'white'}),
    html.Div([
        html.Div([
            html.Button(id='pause-button', n_clicks=0),
            html.Div(id='position-info', style={'color': 'white', 'fontSize': '18px', 'lineHeight': '1.6'}),
            html.Img(src="/video_feed", style={'width': '100%', 'height': '240px', 'marginTop': '20px', 'border': '2px solid white'})
        ], style={'width': '320px', 'padding': '20px'}),
        html.Div([
            dcc.Graph(
                id='3d-plot',
                style={'height': '100%', 'width': '100%'},
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'editable': False,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['orbitRotation', 'zoom3d', 'pan3d']
                }
            ),
            html.Div(id='status-bar', style={
            'position': 'absolute',
            'top': '120px',   # 往下移一点，避免与按钮重叠
            'right': '20px',
            'zIndex': '10',
            'border': '1px solid white',     # 添加白色边框
            'borderRadius': '6px',
            'padding': '6px 10px',
            'backgroundColor': 'rgba(0,0,0,0.6)'  # 半透明黑底更好看
            })
        ], style={'flexGrow': '1', 'overflow': 'hidden', 'height': 'calc(100vh - 80px)', 'position': 'relative'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    dcc.Interval(id='interval', interval=500, n_intervals=0),
    dcc.Store(id='pause-store', data=True)
], style={'backgroundColor': 'black', 'height': '100vh', 'margin': '0', 'padding': '0', 'overflow': 'hidden', 'display': 'flex', 'flexDirection': 'column'})

@app.callback(
    [Output('pause-store', 'data'), Output('pause-button', 'children'), Output('pause-button', 'style')],
    Input('pause-button', 'n_clicks'),
    State('pause-store', 'data')
)
def toggle_pause(n_clicks, is_active):
    is_active = not is_active
    button_text = "Monitoring" if is_active else "Monitor"
    button_style = {
        'marginBottom': '20px',
        'padding': '10px 20px',
        'color': 'white',
        'border': '2px solid white',
        'backgroundColor': '#e74c3c' if is_active else 'transparent',
        'fontWeight': 'bold',
        'fontSize': '16px',
        'letterSpacing': '1px',
        'textTransform': 'uppercase'
    }
    return is_active, button_text, button_style

@app.callback(
    Output('status-bar', 'children'),
    Input('interval', 'n_intervals')
)
def update_status(n):
    with data_lock:
        ok = update_enabled

    color = 'limegreen' if ok else 'red'
    text = 'RTSP Stream OK' if ok else 'RTSP Stream Lost'
    return html.Div([
        html.Div(style={
            'width': '14px', 'height': '14px', 'borderRadius': '50%',
            'backgroundColor': color, 'display': 'inline-block',
            'marginRight': '10px', 'verticalAlign': 'middle'
        }),
        html.Span(text, style={'color': 'white', 'fontSize': '14px'})
    ], style={'display': 'flex', 'alignItems': 'center'})

@app.callback(
    [Output('3d-plot', 'figure'), Output('position-info', 'children')],
    Input('interval', 'n_intervals'),
    State('pause-store', 'data')
)
def update_plot(n, is_active):
    if not is_active:
        raise dash.exceptions.PreventUpdate

    with data_lock:
        global smoothed_pos, robot_positions

        traces = []
        position_display = []

        traces.append(go.Scatter3d(
            x=[smoothed_pos[0]],
            y=[smoothed_pos[1]],
            z=[smoothed_pos[2] + 0.01],
            mode='markers',
            marker=dict(size=8, color='green', sizemode='diameter'),
            hovertext=[f"Camera\n(x={smoothed_pos[0]:.2f}m, y={smoothed_pos[1]:.2f}m, z={smoothed_pos[2]:.2f}m)"],
            hoverinfo='text',
            name='Camera (Boat)'
        ))

        position_display.append(html.Div([
            html.Strong("Camera (Boat):"),
            html.Div(f"x = {smoothed_pos[0]:.2f} m, y = {smoothed_pos[1]:.2f} m, z = {smoothed_pos[2]:.2f} m")
        ], style={'color': 'green', 'marginBottom': '10px'}))

        for rid, pos in robot_positions:
            traces.append(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2] + 0.01],
                mode='markers',
                marker=dict(size=6, color='blue', sizemode='diameter'),
                hovertext=[f"ID {rid}\n(x={pos[0]:.2f}m, y={pos[1]:.2f}m, z={pos[2]:.2f}m)"],
                hoverinfo='text',
                name=f'Robot ID {rid}'
            ))

            position_display.append(html.Div([
                html.Strong(f"Robot ID {rid}:", style={'color': 'dodgerblue'}),
                html.Div(f"x = {pos[0]:.2f} m, y = {pos[1]:.2f} m, z = {pos[2]:.2f} m", style={'color': 'dodgerblue'})
            ], style={'marginBottom': '10px'}))

        fov_angle = np.radians(110)
        cone_height = 0.8
        cone_radius = cone_height * np.tan(fov_angle / 2)
        steps = 40
        theta = np.linspace(0, 2 * np.pi, steps)
        x_base = smoothed_pos[0] + cone_radius * np.cos(theta)
        y_base = smoothed_pos[1] + cone_radius * np.sin(theta)
        z_base = np.full_like(x_base, smoothed_pos[2] - cone_height)

        x_mesh = list(x_base) + [smoothed_pos[0]]
        y_mesh = list(y_base) + [smoothed_pos[1]]
        z_mesh = list(z_base) + [smoothed_pos[2]]

        i, j, k = [], [], []
        for t in range(steps - 1):
            i.append(steps)
            j.append(t)
            k.append(t + 1)
        i.append(steps)
        j.append(steps - 1)
        k.append(0)

        traces.append(go.Mesh3d(
            x=x_mesh, y=y_mesh, z=z_mesh,
            i=i, j=j, k=k,
            color='lightgreen', opacity=0.4,
            name='Camera FOV'
        ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-0.6, 0.6]),
                yaxis=dict(range=[-0.6, 0.6]),
                zaxis=dict(range=[-0.1, 1.1]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1.2),
                camera=dict(eye=dict(x=1.3, y=1.3, z=1.1))
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white'),
            uirevision='constant',
            margin=dict(l=20, r=20, b=10, t=40)
        )
        return fig, position_display

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:8050")
    app.run(debug=False)
