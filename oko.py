import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import time
import yaml
import os
from typing import Dict, Any
from scipy.spatial.transform import Rotation as R


class ArucoDetector:
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.marker_length = float(self.config.get('marker_length', 0.05))
        aruco_dict_name = self.config.get('aruco_dict', 'DICT_5X5_50')
        self.camera_id = int(self.config.get('camera_id', 0))

        if not hasattr(cv2.aruco, aruco_dict_name):
            raise ValueError(f"Неизвестный словарь ArUco: {aruco_dict_name}")
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))

        self.detector_params = cv2.aruco.DetectorParameters()
        det_params_config = self.config.get('detector_params', {})
        for param_name, value in det_params_config.items():
            if hasattr(self.detector_params, param_name):
                setattr(self.detector_params, param_name, value)
            else:
                print(f"Неизвестный параметр детектора: '{param_name}'")

        #Автоматический путь к калибровке в той же папке, что и config.yaml
        config_dir = os.path.dirname(os.path.abspath(config_path))
        calibration_file = os.path.join(config_dir, "camera_calibration_good.npz")

        if not os.path.exists(calibration_file):
            raise FileNotFoundError(f"Файл калибровки не найден: {calibration_file}")

        with np.load(calibration_file) as data:  # ← ВАЖНО: 'data' после 'as'
            self.camera_matrix = data['camera_matrix'].astype(np.float32)
            self.dist_coeffs = data['dist_coeffs'].astype(np.float32)
            if self.dist_coeffs.ndim == 2:
                self.dist_coeffs = self.dist_coeffs.flatten()

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

        half = self.marker_length / 2
        self.marker_obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

        self.last_pose: Dict[int, Dict[str, Any]] = {}

    def detect(self, image: np.ndarray):
        if image is None or image.size == 0:
            return [], None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def estimate_pose(self, corners, ids):
        if ids is None or len(ids) == 0 or len(corners) != len(ids):
            return [], []

        rvecs, tvecs = [], []
        for corner in corners:
            img_points = corner.reshape((4, 2))
            success, rvec, tvec = cv2.solvePnP(
                self.marker_obj_points,
                img_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
            else:
                rvecs.append(np.zeros((3, 1), dtype=np.float32))
                tvecs.append(np.zeros((3, 1), dtype=np.float32))
        return rvecs, tvecs

    def get_pos(self, image: np.ndarray) -> Dict[int, Dict[str, Any]]:
        corners, ids = self.detect(image)
        rvecs, tvecs = self.estimate_pose(corners, ids)

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                marker_id = int(mid)
                rvec = rvecs[i].copy()
                tvec = tvecs[i].copy()
                rmat, _ = cv2.Rodrigues(rvec)
                rot = R.from_matrix(rmat)
                euler = rot.as_euler('xyz', degrees=True)

                self.last_pose[marker_id] = {
                    'position': tvec.flatten(),
                    'distance': float(np.linalg.norm(tvec)),
                    'euler_angles': euler.tolist(),
                    'rotation_vector': rvec.flatten().tolist(),
                    'valid': True
                }

        detected_ids = set(ids.flatten()) if ids is not None else set()
        for mid in self.last_pose:
            if mid not in detected_ids:
                self.last_pose[mid]['valid'] = False

        return self.last_pose.copy()

    def draw(self, image: np.ndarray) -> np.ndarray:
        corners, ids = self.detect(image)
        rvecs, tvecs = self.estimate_pose(corners, ids)
        output = image.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(output, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i].reshape((4, 2))
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))

                pose = self.last_pose.get(int(marker_id))
                if not pose:
                    continue

                dist = pose['distance']
                euler = pose['euler_angles']
                label = f"ID{marker_id}\nD={dist:.2f}m\nR={euler[0]:.0f}° P={euler[1]:.0f}° Y={euler[2]:.0f}°"
                y0 = cy - 60
                for j, line in enumerate(label.split('\n')):
                    cv2.putText(output, line, (cx - 30, y0 + j * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return output


class ColorCalibrator:
    def __init__(self):
        self.calibration_active = False
        self.cap = None
        self.current_hsv_ranges = {
            'h_min': 0, 's_min': 68, 'v_min': 0,
            'h_max': 255, 's_max': 255, 'v_max': 189
        }
    
    def start_calibration(self):
        self.calibration_active = True
        self.cap = cv2.VideoCapture(1)
        threading.Thread(target=self.calibration_loop, daemon=True).start()
    
    def stop_calibration(self):
        self.calibration_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyWindow("result")
        cv2.destroyWindow("settings")
    
    def calibration_loop(self):
        def nothing(*arg):
            pass

        cv2.namedWindow("result")  # создаем главное окно
        cv2.namedWindow("settings")  # создаем окно настроек

        # создаем 6 бегунков для настройки начального и конечного цвета фильтра
        cv2.createTrackbar('h_min', 'settings', self.current_hsv_ranges['h_min'], 255, nothing)
        cv2.createTrackbar('s_min', 'settings', self.current_hsv_ranges['s_min'], 255, nothing)
        cv2.createTrackbar('v_min', 'settings', self.current_hsv_ranges['v_min'], 255, nothing)
        cv2.createTrackbar('h_max', 'settings', self.current_hsv_ranges['h_max'], 255, nothing)
        cv2.createTrackbar('s_max', 'settings', self.current_hsv_ranges['s_max'], 255, nothing)
        cv2.createTrackbar('v_max', 'settings', self.current_hsv_ranges['v_max'], 255, nothing)

        while self.calibration_active:
            flag, img = self.cap.read()
            if not flag:
                break
                
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # считываем значения бегунков
            h1 = cv2.getTrackbarPos('h_min', 'settings')
            s1 = cv2.getTrackbarPos('s_min', 'settings')
            v1 = cv2.getTrackbarPos('v_min', 'settings')
            h2 = cv2.getTrackbarPos('h_max', 'settings')
            s2 = cv2.getTrackbarPos('s_max', 'settings')
            v2 = cv2.getTrackbarPos('v_max', 'settings')

            # сохраняем текущие значения
            self.current_hsv_ranges = {
                'h_min': h1, 's_min': s1, 'v_min': v1,
                'h_max': h2, 's_max': s2, 'v_max': v2
            }

            # формируем начальный и конечный цвет фильтра
            h_min = np.array((h1, s1, v1), np.uint8)
            h_max = np.array((h2, s2, v2), np.uint8)

            # накладываем фильтр на кадр в модели HSV
            thresh = cv2.inRange(hsv, h_min, h_max)
            col = cv2.bitwise_and(hsv, hsv, mask=thresh)
            res = cv2.cvtColor(col, cv2.COLOR_HSV2BGR, col)

            cv2.imshow('result', thresh)
            cv2.imshow('mask', res)

            ch = cv2.waitKey(5)
            if ch == 27:
                break

        self.stop_calibration()
    
    def get_hsv_ranges(self):
        return self.current_hsv_ranges.copy()


class AllSeeingEye:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Всевидящее Око")
        self.root.geometry("1200x800")

        # Инициализация ArUco детектора
        self.aruco_detector = None
        self.aruco_enabled = False
        
        # Инициализация калибратора цвета
        self.color_calibrator = ColorCalibrator()
        self.color_detection_enabled = False
        
        # Переменные
        self.current_mode = "camera"
        self.camera_active = False
        self.cap = None
        self.current_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Панель управления
        control_frame = ttk.LabelFrame(main_frame, text="Управление", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Выбор режима
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X)
        
        ttk.Label(mode_frame, text="Режим:").pack(side=tk.LEFT)
        
        self.mode_var = tk.StringVar(value="camera")
        ttk.Radiobutton(mode_frame, text="Камера", variable=self.mode_var, 
                       value="camera", command=self.switch_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Изображение", variable=self.mode_var, 
                       value="image", command=self.switch_mode).pack(side=tk.LEFT, padx=5)
        
        # Кнопки управления
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Загрузить изображение", 
                  command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Старт камеры", 
                  command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Стоп камеры", 
                  command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        
        # ArUco управление
        aruco_frame = ttk.Frame(control_frame)
        aruco_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(aruco_frame, text="Включить ArUco", 
                  command=self.enable_aruco).pack(side=tk.LEFT, padx=5)
        ttk.Button(aruco_frame, text="Выключить ArUco", 
                  command=self.disable_aruco).pack(side=tk.LEFT, padx=5)
        
        # Управление цветовой калибровкой
        color_frame = ttk.Frame(control_frame)
        color_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(color_frame, text="Калибровка цвета", 
                  command=self.start_color_calibration).pack(side=tk.LEFT, padx=5)
        ttk.Button(color_frame, text="Включить детекцию цвета", 
                  command=self.enable_color_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(color_frame, text="Выключить детекцию цвета", 
                  command=self.disable_color_detection).pack(side=tk.LEFT, padx=5)
        
        # Область отображения
        display_frame = ttk.LabelFrame(main_frame, text="Видео/Изображение", padding=10)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas для отображения
        self.canvas = tk.Canvas(display_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Панель информации
        info_frame = ttk.LabelFrame(main_frame, text="Координаты и информация", padding=10)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def switch_mode(self):
        self.current_mode = self.mode_var.get()
        if self.current_mode == "image":
            self.stop_camera()
    
    def enable_aruco(self):
        try:
            CONFIG_PATH = r"C:\Users\student\Desktop\artHzBan\config.yaml"
            self.aruco_detector = ArucoDetector(CONFIG_PATH)
            self.aruco_enabled = True
            self.info_text.insert(tk.END, "✅ ArUco детектор включен\n")
        except Exception as e:
            self.info_text.insert(tk.END, f"❌ Ошибка инициализации ArUco: {e}\n")
    
    def disable_aruco(self):
        self.aruco_enabled = False
        self.aruco_detector = None
        self.info_text.insert(tk.END, "⏹️ ArUco детектор выключен\n")
    
    def start_color_calibration(self):
        self.color_calibrator.start_calibration()
        self.info_text.insert(tk.END, "🎨 Запущена калибровка цвета\n")
    
    def enable_color_detection(self):
        self.color_detection_enabled = True
        hsv_ranges = self.color_calibrator.get_hsv_ranges()
        self.info_text.insert(tk.END, f"🎨 Детекция цвета включена. Диапазон HSV: {hsv_ranges}\n")
    
    def disable_color_detection(self):
        self.color_detection_enabled = False
        self.info_text.insert(tk.END, "🎨 Детекция цвета выключена\n")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.process_single_image(file_path)
    
    def process_single_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                processed_image = self.process_frame(image_rgb)
                self.display_image(processed_image)
        except Exception as e:
            self.update_info({}, f"Ошибка: {str(e)}")
    
    def start_camera(self):
        if not self.camera_active:
            self.camera_active = True
            camera_id = 0
            if self.aruco_detector:
                camera_id = self.aruco_detector.camera_id
            self.cap = cv2.VideoCapture(camera_id)
            threading.Thread(target=self.camera_loop, daemon=True).start()
    
    def stop_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def camera_loop(self):
        while self.camera_active and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = self.process_frame(frame_rgb)
                self.display_image(processed_frame)
            time.sleep(0.03)
    
    def process_frame(self, image):
        # Копируем изображение для обработки
        processed_image = image.copy()
        all_coordinates = {}
        
        # Обработка MediaPipe
        coordinates = self.detect_mediapipe_features(processed_image)
        all_coordinates.update(coordinates)
        
        # Обработка ArUco если включено
        if self.aruco_enabled and self.aruco_detector:
            aruco_coordinates = self.detect_aruco_features(processed_image)
            all_coordinates.update(aruco_coordinates)
        
        # Обработка цвета если включено
        if self.color_detection_enabled:
            color_coordinates = self.detect_color_features(processed_image)
            all_coordinates.update(color_coordinates)
        
        self.update_info(all_coordinates, self.current_mode)
        return processed_image

    def detect_aruco_features(self, image):
        coordinates = {}
        if self.aruco_detector:
            # Конвертируем обратно в BGR для ArUco
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            aruco_poses = self.aruco_detector.get_pos(image_bgr)
            image_processed = self.aruco_detector.draw(image_bgr)
            # Конвертируем обратно в RGB для отображения
            image_rgb_processed = cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB)
            image[:, :] = image_rgb_processed[:, :]
            
            coordinates['aruco'] = aruco_poses
        
        return coordinates
    
    def detect_color_features(self, image):
        coordinates = {}
        # Конвертируем в HSV для цветовой детекции
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Получаем текущие диапазоны HSV
        hsv_ranges = self.color_calibrator.get_hsv_ranges()
        h_min = np.array((hsv_ranges['h_min'], hsv_ranges['s_min'], hsv_ranges['v_min']), np.uint8)
        h_max = np.array((hsv_ranges['h_max'], hsv_ranges['s_max'], hsv_ranges['v_max']), np.uint8)
        
        # Применяем цветовой фильтр
        mask = cv2.inRange(hsv, h_min, h_max)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color_objects = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Фильтр по площади
                x, y, w, h = cv2.boundingRect(contour)
                # Рисуем bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(image, "Color Object", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                color_objects.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'area': cv2.contourArea(contour),
                    'center': (x + w//2, y + h//2)
                })
        
        if color_objects:
            coordinates['color_objects'] = color_objects
        
        return coordinates
    
    def extract_landmarks(self, landmarks, width, height):
        coords = {}
        for idx, landmark in enumerate(landmarks.landmark):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            coords[idx] = {'x': x, 'y': y, 'z': landmark.z}
        return coords
    
    def extract_hand_landmarks(self, multi_hand_landmarks, width, height):
        hands_coords = []
        for hand_landmarks in multi_hand_landmarks:
            hand_coords = {}
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                hand_coords[idx] = {'x': x, 'y': y, 'z': landmark.z}
            hands_coords.append(hand_coords)
        return hands_coords
    
    def display_image(self, image):
        try:
            # Масштабируем изображение под размер canvas
            h, w = image.shape[:2]
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                scale = min(canvas_width / w, canvas_height / h) * 0.95
                new_w, new_h = int(w * scale), int(h * scale)
                image_resized = cv2.resize(image, (new_w, new_h))
                
                # Конвертируем для tkinter
                photo = ImageTk.PhotoImage(image=Image.fromarray(image_resized))
                
                # Обновляем canvas
                self.canvas.delete("all")
                self.canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                       image=photo, anchor=tk.CENTER)
                self.canvas.image = photo
        except Exception as e:
            print(f"Ошибка отображения: {e}")
    
    def update_info(self, coordinates, source):
        info_text = f"Источник: {source}\n"
        info_text += f"Время: {time.strftime('%H:%M:%S')}\n"
        info_text += f"ArUco: {'ВКЛ' if self.aruco_enabled else 'ВЫКЛ'}\n"
        info_text += f"Детекция цвета: {'ВКЛ' if self.color_detection_enabled else 'ВЫКЛ'}\n\n"
        
        if 'pose' in coordinates:
            info_text += "=== ПОЗА ===\n"
            for idx, coord in coordinates['pose'].items():
                info_text += f"Точка {idx}: ({coord['x']}, {coord['y']}, {coord['z']:.3f})\n"
        
        if 'hands' in coordinates:
            info_text += f"\n=== РУКИ (найдено: {len(coordinates['hands'])} ) ===\n"
            for i, hand in enumerate(coordinates['hands']):
                info_text += f"Рука {i+1}:\n"
                for idx, coord in list(hand.items())[:5]:  # Показываем первые 5 точек
                    info_text += f"  Точка {idx}: ({coord['x']}, {coord['y']})\n"
        
        if 'faces' in coordinates:
            info_text += f"\n=== ЛИЦА (найдено: {len(coordinates['faces'])} ) ===\n"
            for i, face in enumerate(coordinates['faces']):
                info_text += f"Лицо {i+1}: x={face['x']}, y={face['y']}, w={face['width']}, h={face['height']}\n"
        
        if 'aruco' in coordinates:
            info_text += f"\n=== ARUCO МАРКЕРЫ ===\n"
            for marker_id, data in coordinates['aruco'].items():
                status = "Виден" if data['valid'] else "Потерян"
                pos = data['position']
                e = data['euler_angles']
                dist = data['distance']
                info_text += f"Маркер ID{marker_id} [{status}]:\n"
                info_text += f"  Расстояние: {dist:.3f} м\n"
                info_text += f"  Углы (R,P,Y): [{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}]°\n"
                info_text += f"  Позиция (X,Y,Z): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
        
        if 'color_objects' in coordinates:
            info_text += f"\n=== ОБЪЕКТЫ ПО ЦВЕТУ (найдено: {len(coordinates['color_objects'])} ) ===\n"
            hsv_ranges = self.color_calibrator.get_hsv_ranges()
            info_text += f"Диапазон HSV: min({hsv_ranges['h_min']};{hsv_ranges['s_min']};{hsv_ranges['v_min']}) "
            info_text += f"max({hsv_ranges['h_max']};{hsv_ranges['s_max']};{hsv_ranges['v_max']})\n"
            for i, obj in enumerate(coordinates['color_objects']):
                info_text += f"Объект {i+1}: x={obj['x']}, y={obj['y']}, w={obj['width']}, h={obj['height']}, "
                info_text += f"площадь={obj['area']:.1f}, центр=({obj['center'][0]}, {obj['center'][1]})\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
    
    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.stop_camera()
            self.color_calibrator.stop_calibration()
            self.pose.close()
            self.hands.close()
            self.face.close()

if __name__ == "__main__":
    app = AllSeeingEye()
    app.run()
