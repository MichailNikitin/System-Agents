from oko import AllSeeEye
import cv2

boss = AllSeeEye(r"C:\Users\Asus\Documents\GitHub\System-Agents-\src\config.yaml")

while True:
    poses = boss.get_pos()
    print(poses)
    if poses:
        boss.draw_marker_quad( marker_ids=[0, 1, 2, 3],
    color=(255, 0, 0),
    thickness=3)
        #boss.debug_draw()



    if cv2.waitKey(1) == 27:  # ESC
        break

