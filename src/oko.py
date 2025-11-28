from Aruca import ArucoDetector
import cv2
import numpy as np
import yaml
import  sys

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

