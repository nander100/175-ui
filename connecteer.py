import sys
import cv2
import numpy as np
import socket
import json
import threading
import struct
import select
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider,
                             QLineEdit, QGroupBox, QGridLayout, QTabWidget)
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


class PiWebcamThread(QThread):
    """
    Receives webcam stream from Raspberry Pi with low-latency optimizations:
    - JPEG decoding (matches server's JPEG encoding)
    - Frame dropping to always display newest frame
    - Non-blocking socket with select()
    - TCP_NODELAY and reduced buffer sizes
    """
    frame_ready = pyqtSignal(np.ndarray)
    connection_status = pyqtSignal(str)

    def __init__(self, pi_ip):
        super().__init__()
        self.pi_ip = pi_ip
        self.running = False
        self.socket = None

    def run(self):
        try:
            self.connection_status.emit("Connecting to Pi webcam...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Optimize socket for low latency
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            self.socket.settimeout(5)
            self.socket.connect((self.pi_ip, 8001))
            self.socket.settimeout(0.1)  # Short timeout for non-blocking reads
            self.connection_status.emit("Pi webcam connected")
            self.running = True

            data_buffer = b""
            payload_size = struct.calcsize("Q")

            while self.running:
                try:
                    # Use select to check if data is available (non-blocking)
                    ready = select.select([self.socket], [], [], 0.01)
                    if not ready[0]:
                        continue

                    # Receive all available data (drain the buffer to get latest frame)
                    while True:
                        try:
                            chunk = self.socket.recv(65536)
                            if not chunk:
                                self.connection_status.emit("Pi webcam disconnected")
                                self.running = False
                                break
                            data_buffer += chunk
                        except socket.timeout:
                            break  # No more data available right now
                        except BlockingIOError:
                            break

                    if not self.running:
                        break

                    # Process all complete frames, keep only the last one
                    last_frame = None
                    while len(data_buffer) >= payload_size:
                        # Extract message size
                        msg_size = struct.unpack("Q", data_buffer[:payload_size])[0]

                        # Check if we have complete frame
                        if len(data_buffer) < payload_size + msg_size:
                            break  # Wait for more data

                        # Extract frame data
                        frame_data = data_buffer[payload_size:payload_size + msg_size]
                        data_buffer = data_buffer[payload_size + msg_size:]

                        # Decode JPEG frame
                        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                        if frame is not None:
                            last_frame = frame

                    # Only emit the most recent frame (drop older ones)
                    if last_frame is not None:
                        self.frame_ready.emit(last_frame)

                except Exception as e:
                    if self.running:
                        self.connection_status.emit(f"Pi webcam error: {e}")
                    break

        except Exception as e:
            self.connection_status.emit(f"Pi webcam error: {e}")
            print(f"Pi webcam error: {e}")
        finally:
            if self.socket:
                self.socket.close()
            self.connection_status.emit("Pi webcam disconnected")

    def stop(self):
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass


class RobotArmController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Arm Controller")
        self.setGeometry(100, 100, 950, 850)

        self.pi_socket = None
        self.camera_thread = None
        self.pi_webcam_thread = None
        self.tracking_mode = False
        self.servo_positions = [0, 50, 90, 90]  # base, arm1, arm2, gripper
        self.speed = 50  # Default speed

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

        # Tab widget for camera feeds
        self.tab_widget = QTabWidget()

        # RealSense tab
        realsense_widget = QWidget()
        realsense_layout = QVBoxLayout(realsense_widget)
        self.camera_label = QLabel("Click 'Start RealSense' to begin")
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        realsense_layout.addWidget(self.camera_label)

        # Hand info
        self.hand_info = QLabel("Hand Position: N/A | Finger Bend: N/A")
        realsense_layout.addWidget(self.hand_info)

        self.tab_widget.addTab(realsense_widget, "RealSense Camera")

        # Pi Webcam tab
        pi_webcam_widget = QWidget()
        pi_webcam_layout = QVBoxLayout(pi_webcam_widget)
        self.pi_webcam_label = QLabel("Connect to Pi and click 'Start Pi Webcam'")
        self.pi_webcam_label.setMinimumSize(640, 480)
        self.pi_webcam_label.setStyleSheet("border: 2px solid blue;")
        self.pi_webcam_label.setAlignment(Qt.AlignCenter)
        pi_webcam_layout.addWidget(self.pi_webcam_label)

        # Webcam status
        self.webcam_status_label = QLabel("Status: Not connected")
        pi_webcam_layout.addWidget(self.webcam_status_label)

        self.tab_widget.addTab(pi_webcam_widget, "Pi Webcam")

        layout.addWidget(self.tab_widget)

        # Controls
        ctrl_layout = QHBoxLayout()
        self.camera_btn = QPushButton("Start RealSense")
        self.camera_btn.clicked.connect(self.toggle_camera)
        ctrl_layout.addWidget(self.camera_btn)

        self.pi_webcam_btn = QPushButton("Start Pi Webcam")
        self.pi_webcam_btn.clicked.connect(self.toggle_pi_webcam)
        self.pi_webcam_btn.setEnabled(False)
        ctrl_layout.addWidget(self.pi_webcam_btn)

        self.track_btn = QPushButton("Start Tracking Mode")
        self.track_btn.clicked.connect(self.toggle_tracking)
        self.track_btn.setEnabled(False)
        ctrl_layout.addWidget(self.track_btn)
        layout.addLayout(ctrl_layout)

        # Speed control
        speed_group = QGroupBox("Movement Speed")
        speed_layout = QHBoxLayout(speed_group)
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.speed_changed)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("50")
        speed_layout.addWidget(self.speed_label)
        layout.addWidget(speed_group)

        # Servos
        servo_group = QGroupBox("Manual Servo Control")
        servo_layout = QGridLayout(servo_group)
        self.sliders = []
        self.labels = []

        servo_names = ["Base (0-90°)", "Arm1 (50-100°)", "Arm2 (90-150°)", "Gripper (90-160°)"]
        servo_ranges = [(0, 90), (50, 100), (90, 150), (90, 160)]
        servo_defaults = [0, 50, 90, 90]

        for i in range(4):
            servo_layout.addWidget(QLabel(servo_names[i]), i, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(servo_ranges[i][0], servo_ranges[i][1])
            slider.setValue(servo_defaults[i])
            slider.valueChanged.connect(lambda v, idx=i: self.servo_changed(idx, v))
            servo_layout.addWidget(slider, i, 1)
            label = QLabel(f"{servo_defaults[i]}°")
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
            self.pi_webcam_btn.setEnabled(False)
        else:
            try:
                self.pi_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.pi_socket.connect((self.ip_input.text(), 8000))
                self.connect_btn.setText("Disconnect")
                self.status_label.setText("Connected")
                self.track_btn.setEnabled(True)
                self.pi_webcam_btn.setEnabled(True)
            except Exception as e:
                self.status_label.setText(f"Error: {e}")

    def toggle_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_btn.setText("Start RealSense")
        else:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.hand_data.connect(self.update_hand_data)
            self.camera_thread.start()
            self.camera_btn.setText("Stop RealSense")

    def toggle_pi_webcam(self):
        if self.pi_webcam_thread and self.pi_webcam_thread.isRunning():
            self.pi_webcam_thread.stop()
            self.pi_webcam_thread.wait()
            self.pi_webcam_btn.setText("Start Pi Webcam")
            self.webcam_status_label.setText("Status: Stopped")
        else:
            self.pi_webcam_thread = PiWebcamThread(self.ip_input.text())
            self.pi_webcam_thread.frame_ready.connect(self.update_pi_webcam_frame)
            self.pi_webcam_thread.connection_status.connect(self.update_webcam_status)
            self.pi_webcam_thread.start()
            self.pi_webcam_btn.setText("Stop Pi Webcam")

    def toggle_tracking(self):
        self.tracking_mode = not self.tracking_mode
        self.track_btn.setText("Stop Tracking" if self.tracking_mode else "Start Tracking Mode")
        for slider in self.sliders:
            slider.setEnabled(not self.tracking_mode)

    def speed_changed(self, value):
        self.speed = value
        self.speed_label.setText(str(value))

    def update_frame(self, frame):
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_img.rgbSwapped()).scaled(
            640, 480, Qt.KeepAspectRatio))

    def update_pi_webcam_frame(self, frame):
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.pi_webcam_label.setPixmap(QPixmap.fromImage(qt_img.rgbSwapped()).scaled(
            640, 480, Qt.KeepAspectRatio))

    def update_webcam_status(self, status):
        self.webcam_status_label.setText(f"Status: {status}")

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
            cmd = {
                'type': 'servo_control',
                'positions': {f'servo{i}': self.servo_positions[i] for i in range(4)},
                'speed': self.speed
            }
            try:
                self.pi_socket.sendall((json.dumps(cmd) + '\n').encode())
            except:
                pass

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
        if self.pi_webcam_thread:
            self.pi_webcam_thread.stop()
            self.pi_webcam_thread.wait()
        if self.pi_socket:
            self.pi_socket.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RobotArmController()
    window.show()
    sys.exit(app.exec_())
