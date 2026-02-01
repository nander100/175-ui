
# # import socket
# # import json
# # import time
# # import serial


# # def set_servo_angle(channel, angle): # jon is poop
# #     """Simulate servo angle command (no hardware needed)"""
# #     print(f"[TEST] Servo {channel} -> {angle}°")
    
# #     # In real version, this would be: ser.write(f"{channel},{angle}\n".encode())

# # # Create socket server
# # server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# # server.bind(('0.0.0.0', 8000))
# # server.listen(1)

# # # Simulated serial connection 
# # ser = serial.Serial(
# #     port='/dev/ttyUSB0',
# #     baudrate=9600,
# #     timeout=1
# # )

# # print("=" * 50)
# # print("TEST SERVER - Running on port 8000")
# # print("Waiting for connections...")
# # print("=" * 50)

# # while True:
# #     try:
# #         client, addr = server.accept()
# #         print(f"\n Connected to {addr}")
# #         print("-" * 50)
        
# #         buffer = ""
# #         while True:
# #             data = client.recv(1024).decode('utf-8')
# #             if not data:
# #                 print(f"Client {addr} disconnected")
# #                 break
            
# #             buffer += data
            
# #             # Process complete JSON commands (separated by newlines)
# #             while '\n' in buffer:
# #                 line, buffer = buffer.split('\n', 1)
# #                 if not line.strip():
# #                     continue
                
# #                 try:
# #                     print(f"\nReceived: {line}")
# #                     cmd = json.loads(line)
                    
# #                     if cmd['type'] == 'servo_control':
# #                         positions = cmd['positions']
# #                         print(f"Command type: servo_control")
# #                         print(f"Positions: {positions}")
                        
# #                         for i in range(4):
# #                             if f'servo{i}' in positions:
# #                                 angle = positions[f'servo{i}']
# #                                 set_servo_angle(i, angle)
                                
# #                     print("-" * 50)
                    
# #                 except json.JSONDecodeError as e:
# #                     print(f"✗ JSON Error: {e}")
# #                 except Exception as e:
# #                     print(f"✗ Error processing command: {e}")
                    
# #     except KeyboardInterrupt:
# #         print("\n\nShutting down server...")
# #         break
# #     except Exception as e:
# #         print(f" Connection error: {e}")
# #     finally:
# #         try:
# #             client.close()
# #         except:
# #             pass

# # server.close()
# # print("Server stopped.")    

# import socket
# import json
# import time
# import serial

# # Open serial connection to Arduino
# ser = serial.Serial(
#     port='/dev/ttyUSB0',
#     baudrate=9600,
#     timeout=1
# )

# time.sleep(2)  # Wait for Arduino to initialize

# def move_arm(base, arm1, arm2, gripper, speed):
#     """
#     Move robot arm to specified positions
    
#     Args:
#         base: 0-90 degrees
#         arm1: 50-100 degrees
#         arm2: 90-150 degrees
#         gripper: 90-160 degrees
#         speed: 1-100 (1=slow, 100=fast)
#     """
#     command = f"{base},{arm1},{arm2},{gripper},{speed}\n"
#     print(f"Sending: {command.strip()}")
    
#     # Clear any old data in buffer
#     ser.reset_input_buffer()
    
#     # Send command
#     ser.write(command.encode())
    
#     # Wait for responses
#     time.sleep(0.1)
#     while ser.in_waiting > 0:
#         response = ser.readline().decode().strip()
#         print(f"Arduino: {response}")

# # Create socket server
# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# server.bind(('0.0.0.0', 8000))
# server.listen(1)

# print("=" * 50)
# print("SERVER - Running on port 8000")
# print("Waiting for connections...")
# print("=" * 50)

# # Track current positions
# current_positions = {
#     'base': 0,
#     'arm1': 50,
#     'arm2': 90,
#     'gripper': 90
# }

# while True:
#     try:
#         client, addr = server.accept()
#         print(f"\n✓ Connected to {addr}")
#         print("-" * 50)
        
#         buffer = ""
#         while True:
#             data = client.recv(1024).decode('utf-8')
#             if not data:
#                 print(f"✗ Client {addr} disconnected")
#                 break
            
#             buffer += data
            
#             # Process complete JSON commands (separated by newlines)
#             while '\n' in buffer:
#                 line, buffer = buffer.split('\n', 1)
#                 if not line.strip():
#                     continue
                
#                 try:
#                     print(f"\nReceived: {line}")
#                     cmd = json.loads(line)
                    
#                     if cmd['type'] == 'servo_control':
#                         positions = cmd['positions']
#                         print(f"Command type: servo_control")
#                         print(f"Positions: {positions}")
                        
#                         # Map servo0-3 to base, arm1, arm2, gripper
#                         servo_map = {
#                             'servo0': 'base',
#                             'servo1': 'arm1',
#                             'servo2': 'arm2',
#                             'servo3': 'gripper'
#                         }
                        
#                         # Update current positions
#                         for servo_key, arm_key in servo_map.items():
#                             if servo_key in positions:
#                                 current_positions[arm_key] = positions[servo_key]
                        
#                         # Get speed from command or use default
#                         speed = cmd.get('speed', 50)  # Default speed: 50
                        
#                         # Send command to Arduino
#                         move_arm(
#                             base=current_positions['base'],
#                             arm1=current_positions['arm1'],
#                             arm2=current_positions['arm2'],
#                             gripper=current_positions['gripper'],
#                             speed=speed
#                         )
                                
#                     print("-" * 50)
                    
#                 except json.JSONDecodeError as e:
#                     print(f"✗ JSON Error: {e}")
#                 except Exception as e:
#                     print(f"✗ Error processing command: {e}")
                    
#     except KeyboardInterrupt:
#         print("\n\nShutting down server...")
#         break
#     except Exception as e:
#         print(f"✗ Connection error: {e}")
#     finally:
#         try:
#             client.close()
#         except:
#             pass

# ser.close()
# server.close()
# print("Server stopped.")

import socket
import json
import time
import serial
import cv2
import pickle
import struct
import threading

# Open serial connection to Arduino
ser = serial.Serial(
    port='/dev/ttyUSB0',
    baudrate=9600,
    timeout=1
)

time.sleep(2)  # Wait for Arduino to initialize

def move_arm(base, arm1, arm2, gripper, speed):
    """
    Move robot arm to specified positions
    
    Args:
        base: 0-90 degrees
        arm1: 50-100 degrees
        arm2: 90-150 degrees
        gripper: 90-160 degrees
        speed: 1-100 (1=slow, 100=fast)
    """
    command = f"{base},{arm1},{arm2},{gripper},{speed}\n"
    print(f"Sending: {command.strip()}")
    
    # Clear any old data in buffer
    ser.reset_input_buffer()
    
    # Send command
    ser.write(command.encode())
    
    # Wait for responses
    time.sleep(0.1)
    while ser.in_waiting > 0:
        response = ser.readline().decode().strip()
        print(f"Arduino: {response}")


def webcam_server():
    """
    Separate server for streaming webcam feed
    Runs on port 8001

    Optimized for low latency:
    - Uses JPEG compression instead of pickle (10-50x smaller)
    - TCP_NODELAY to disable Nagle's algorithm
    - No artificial delays
    - Reduced buffer sizes
    """
    print("\n[WEBCAM] Starting webcam server on port 8001...")

    # Create socket for video streaming
    video_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    video_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    video_server.bind(('0.0.0.0', 8001))
    video_server.listen(5)

    print("[WEBCAM] Webcam server ready, waiting for connections...")

    while True:
        try:
            client_socket, addr = video_server.accept()
            print(f"[WEBCAM] Client connected from {addr}")

            # Optimize socket for low latency
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Reduce send buffer to prevent frame accumulation
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

            # Start webcam capture
            cap = cv2.VideoCapture(0)  # USB webcam (use 0 for default, or try 1, 2 if needed)

            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            # Reduce camera buffer to get freshest frames
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print("[WEBCAM] Error: Could not open webcam")
                client_socket.close()
                continue

            print("[WEBCAM] Webcam opened successfully, streaming with JPEG compression...")

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("[WEBCAM] Failed to capture frame")
                        break

                    # Encode frame as JPEG (much smaller than pickle)
                    # Quality 80 gives good balance of size vs quality
                    _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    data = encoded.tobytes()

                    # Send message size and frame data
                    message_size = struct.pack("Q", len(data))
                    try:
                        client_socket.sendall(message_size + data)
                    except:
                        print("[WEBCAM] Client disconnected")
                        break

                    # No sleep - let camera capture rate control timing
                    # This prevents frame accumulation and reduces latency

            except Exception as e:
                print(f"[WEBCAM] Error during streaming: {e}")
            finally:
                cap.release()
                client_socket.close()
                print("[WEBCAM] Client disconnected, webcam released")

        except Exception as e:
            print(f"[WEBCAM] Connection error: {e}")


# Start webcam server in separate thread
webcam_thread = threading.Thread(target=webcam_server, daemon=True)
webcam_thread.start()

# Create socket server for servo control
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 8000))
server.listen(1)

print("=" * 50)
print("SERVO SERVER - Running on port 8000")
print("Waiting for connections...")
print("=" * 50)

# Track current positions
current_positions = {
    'base': 0,
    'arm1': 50,
    'arm2': 90,
    'gripper': 90
}

while True:
    try:
        client, addr = server.accept()
        print(f"\n✓ Connected to {addr}")
        print("-" * 50)
        
        buffer = ""
        while True:
            data = client.recv(1024).decode('utf-8')
            if not data:
                print(f"✗ Client {addr} disconnected")
                break
            
            buffer += data
            
            # Process complete JSON commands (separated by newlines)
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if not line.strip():
                    continue
                
                try:
                    print(f"\nReceived: {line}")
                    cmd = json.loads(line)
                    
                    if cmd['type'] == 'servo_control':
                        positions = cmd['positions']
                        print(f"Command type: servo_control")
                        print(f"Positions: {positions}")
                        
                        # Map servo0-3 to base, arm1, arm2, gripper
                        servo_map = {
                            'servo0': 'base',
                            'servo1': 'arm1',
                            'servo2': 'arm2',
                            'servo3': 'gripper'
                        }
                        
                        # Update current positions
                        for servo_key, arm_key in servo_map.items():
                            if servo_key in positions:
                                current_positions[arm_key] = positions[servo_key]
                        
                        # Get speed from command or use default
                        speed = cmd.get('speed', 50)  # Default speed: 50
                        
                        # Send command to Arduino
                        move_arm(
                            base=current_positions['base'],
                            arm1=current_positions['arm1'],
                            arm2=current_positions['arm2'],
                            gripper=current_positions['gripper'],
                            speed=speed
                        )
                                
                    print("-" * 50)
                    
                except json.JSONDecodeError as e:
                    print(f"✗ JSON Error: {e}")
                except Exception as e:
                    print(f"✗ Error processing command: {e}")
                    
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        break
    except Exception as e:
        print(f"✗ Connection error: {e}")
    finally:
        try:
            client.close()
        except:
            pass

ser.close()
server.close()
print("Server stopped.")