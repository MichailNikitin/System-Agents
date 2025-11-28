from Aruca import ArucoDetector
import cv2
import numpy as np
import yaml

class AllSeeEye:
    def __init__(self, path_yaml):
        with open("config.yaml", 'r') as file:
            data = yaml.safe_load(file)

        self.id_cam = data['camera_id']
        if data['calibration_file'] is not None:
            self.path2npz = data['calibration_file']
        else:
            print('')
