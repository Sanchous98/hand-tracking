from time import time
from hand_detector import HandDetector
from virtual_mouse import VirtualMouse
import cv2.cv2 as cv

DEBUG = True


def main():
    capture = cv.VideoCapture(0)
    vm = VirtualMouse(capture, HandDetector(max_hands=1), 1280, 720, DEBUG)
    previous_frame_time = time()

    while capture.isOpened():
        image = vm.get_image()

        if DEBUG:
            current_frame_time = time()
            fps = 1 / (current_frame_time - previous_frame_time)
            previous_frame_time = current_frame_time
            cv.putText(image, str(int(fps)), (20, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv.imshow("Image", image)
            cv.waitKey(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
