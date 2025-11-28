from Aruca import ArucoDetector
import cv2
import numpy as np
import yaml
import  sys
from typing import List, Tuple, Optional

class AllSeeEye:
    def __init__(self, path_yaml):
        with open(path_yaml, "r") as file:
            data = yaml.safe_load(file)

        self.camera_id = data["camera_id"]
        self.cap = cv2.VideoCapture(self.camera_id)
        if data["calibration_file"] is not None:
            self.path2npz = data["calibration_file"]
        else:
            print("Warning: Dont have a npz file", file=sys.stderr)
        self.detector = ArucoDetector(path_yaml)

    def get_pos(self):
        ret, frame = self.cap.read()
        all_poses = self.detector.get_pos(frame)
        return all_poses

    def debug_draw(self):
        ret, frame = self.cap.read()
        debug_frame = self.detector.draw(frame)
        cv2.imshow("Aruco Tracker", debug_frame)

    def draw_marker_quad(
            self,
            marker_ids: List[int] = [10, 11, 12, 13],
            color: Tuple[int, int, int] = (0, 255, 0),
            thickness: int = 2
    ) -> np.ndarray:
        """
        Рисует четырёхугольник по центрам четырёх заданных ArUco-маркеров.

        Args:
            image: Исходное изображение (BGR, shape HxWx3).
            marker_ids: Список из 4 ID маркеров по часовой стрелке или в произвольном порядке.
            color: Цвет линии (BGR).
            thickness: Толщина линии.

        Returns:
            Изображение с нарисованным четырёхугольником.
        """
        if len(marker_ids) != 4:
            raise ValueError("marker_ids must contain exactly 4 marker IDs.")
        ret, frame = self.cap.read()
        # Получаем позы маркеров
        poses = self.get_pos()

        # Собираем центры в пикселях
        points = []
        for mid in marker_ids:
            if mid not in poses or not poses[mid].get('valid', False):
                # Если хотя бы один маркер не обнаружен — не рисуем
                return frame.copy()
            center_px = poses[mid].get('center_px')
            if center_px is None:
                return frame.copy()
            points.append(center_px)

        pts = np.array(points, dtype=np.float32)  # shape (4, 2)

        # Сортируем точки в порядке: верхний левый, верхний правый, нижний правый, нижний левый
        # Используем стандартный метод сортировки по сумме и разности координат
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)  # x + y
        diff = np.diff(pts, axis=1)  # y - x

        rect[0] = pts[np.argmin(s)]  # верхний левый (мин x+y)
        rect[2] = pts[np.argmax(s)]  # нижний правый (макс x+y)
        rect[1] = pts[np.argmin(diff)]  # верхний правый (мин y - x → макс x при фикс. y)
        rect[3] = pts[np.argmax(diff)]  # нижний левый (макс y - x)

        # Рисуем замкнутый четырёхугольник
        output_img = frame.copy()
        rect_int = rect.astype(int)
        # Рисуем линии между углами
        for i in range(4):
            cv2.line(
                output_img,
                tuple(rect_int[i]),
                tuple(rect_int[(i + 1) % 4]),
                color,
                thickness
            )
        # Опционально: нарисовать сам четырёхугольник как контур
        # cv2.polylines(output_img, [rect_int], isClosed=True, color=color, thickness=thickness)

        cv2.imshow("Aruco Tracker", output_img)




