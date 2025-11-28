from oko import AllSeeEye
import cv2

boss = AllSeeEye(r"C:\Users\Asus\Documents\GitHub\System-Agents-\src\config.yaml")

while True:
    poses = boss.get_pos()
    print(poses)
    if poses:
        boss.debug_draw()
    if cv2.waitKey(1) == 27:  # ESC
        break
