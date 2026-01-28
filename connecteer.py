import sys
import cv2
import numpy as np
import socket
import json
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QLineEdit, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
import pyrealsense2 as rs
import mediapipe as mp

class CameraThread(QThread):
    """Handles RealSense camera and hand tracking"""
    frame_ready = pyqtSignal(np.ndarray)
    hand_data = pyqtSignal(np.ndarray, float)  # position, bend_angle
    
    def __init__(self):
        super().__init__()
        self.running = False
        
    def run(self):
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils
        
        # Initialize RealSense
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        
        align = rs.align(rs.stream.color)
        self.running = True
        
        while self.running:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Process hand
            results = hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(color_image, hand, mp_hands.HAND_CONNECTIONS)
                
                # Get hand center position
                h, w = color_image.shape[:2]
                cx = int(sum([lm.x for lm in hand.landmark]) / 21 * w)
                cy = int(sum([lm.y for lm in hand.landmark]) / 21 * h)
                
                # Get 3D position
                depth = depth_frame.get_distance(cx, cy)
                intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                point_3d = rs.rs2_deproject_pixel_to_point(intrin, [cx, cy], depth)
                position = np.array([[point_3d[0]], [point_3d[1]], [point_3d[2]]])
                
                # Calculate finger bend (index finger)
                mcp, pip, dip = hand.landmark[5], hand.landmark[6], hand.landmark[7]
                v1 = np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])
                v2 = np.array([dip.x - pip.x, dip.y - pip.y, dip.z - pip.z])
                angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / 
                                   (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
                
                # Draw center point
                cv2.circle(color_image, (cx, cy), 10, (0, 255, 0), -1)
                
                self.hand_data.emit(position, angle)
            
            self.frame_ready.emit(color_image)
        
        hands.close()
        pipeline.stop()

    
        
    def stop(self):
        self.running = False


class RobotArmController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Arm Controller")
        self.setGeometry(100, 100, 900, 700)
        
        self.pi_socket = None
        self.camera_thread = None
        self.tracking_mode = False
        self.servo_positions = [90, 90, 90, 90]
        
        self.init_ui()
        
    def init_ui(self):
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        
        # Connection
        conn_layout = QHBoxLayout()
        conn_layout.addWidget(QLabel("Pi IP:"))
        self.ip_input = QLineEdit("100.71.223.50")
        conn_layout.addWidget(self.ip_input)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        conn_layout.addWidget(self.connect_btn)
        self.status_label = QLabel("Disconnected")
        conn_layout.addWidget(self.status_label)
        layout.addLayout(conn_layout)
        
        # Camera feed
        self.camera_label = QLabel("Click 'Start Camera' to begin")
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label)
        
        # Hand info
        self.hand_info = QLabel("Hand Position: N/A | Finger Bend: N/A")
        layout.addWidget(self.hand_info)
        
        # Controls
        ctrl_layout = QHBoxLayout()
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        ctrl_layout.addWidget(self.camera_btn)
        
        self.track_btn = QPushButton("Start Tracking Mode")
        self.track_btn.clicked.connect(self.toggle_tracking)
        self.track_btn.setEnabled(False)
        ctrl_layout.addWidget(self.track_btn)
        layout.addLayout(ctrl_layout)
        
        # Servos
        servo_group = QGroupBox("Manual Servo Control")
        servo_layout = QGridLayout(servo_group)
        self.sliders = []
        self.labels = []
        
        for i in range(4):
            servo_layout.addWidget(QLabel(f"Servo {i}:"), i, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 180)
            slider.setValue(90)
            slider.valueChanged.connect(lambda v, idx=i: self.servo_changed(idx, v))
            servo_layout.addWidget(slider, i, 1)
            label = QLabel("90°")
            servo_layout.addWidget(label, i, 2)
            self.sliders.append(slider)
            self.labels.append(label)
        
        layout.addWidget(servo_group)
        
    def toggle_connection(self):
        if self.pi_socket:
            self.pi_socket.close()
            self.pi_socket = None
            self.connect_btn.setText("Connect")
            self.status_label.setText("Disconnected")
            self.track_btn.setEnabled(False)
        else:
            try:
                self.pi_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.pi_socket.connect((self.ip_input.text(), 8000))
                self.connect_btn.setText("Disconnect")
                self.status_label.setText("Connected")
                self.track_btn.setEnabled(True)
            except Exception as e:
                self.status_label.setText(f"Error: {e}")
                
    def toggle_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_btn.setText("Start Camera")
        else:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.hand_data.connect(self.update_hand_data)
            self.camera_thread.start()
            self.camera_btn.setText("Stop Camera")
            
    def toggle_tracking(self):
        self.tracking_mode = not self.tracking_mode
        self.track_btn.setText("Stop Tracking" if self.tracking_mode else "Start Tracking Mode")
        for slider in self.sliders:
            slider.setEnabled(not self.tracking_mode)
            
    def update_frame(self, frame):
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_img.rgbSwapped()).scaled(
            640, 480, Qt.KeepAspectRatio))
        
    def update_hand_data(self, position, bend_angle):
        self.hand_info.setText(
            f"Position: X={position[0][0]:.2f}m Y={position[1][0]:.2f}m Z={position[2][0]:.2f}m | "
            f"Finger: {bend_angle:.0f}°"
        )
        
        # TODO: Add your inverse kinematics here to convert position -> servo angles
        # For now, just a placeholder
        if self.tracking_mode:
            pass  # Calculate and send servo positions based on hand position
            
    def servo_changed(self, idx, value):
        self.servo_positions[idx] = value
        self.labels[idx].setText(f"{value}°")
        if not self.tracking_mode:
            self.send_servos()
            
    def send_servos(self):
        if self.pi_socket:
            cmd = {'type': 'servo_control', 
                   'positions': {f'servo{i}': self.servo_positions[i] for i in range(4)}}
            try:
                self.pi_socket.sendall((json.dumps(cmd) + '\n').encode())
            except:
                pass
                
    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
        if self.pi_socket:
            self.pi_socket.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RobotArmController()
    window.show()
    sys.exit(app.exec_())