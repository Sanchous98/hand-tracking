import pyautogui
import cv2.cv2 as cv
from time import time
from typing import Dict, Union
from hand_detector import HandDetector
from virtual_mouse import VirtualMouse, Move, Click, DoubleClick
from event_dispatcher import Listener, Event, EventDispatcher

DEBUG = True
pyautogui.PAUSE = 0


class MouseListener(Listener):
    @property
    def dispatches(self) -> Dict[Union[Event, str], callable]:
        return {
            "mouse.move": self.move,
            "mouse.click": self.click,
            "mouse.dblclick": self.double_click,
        }

    @staticmethod
    def move(event: Move):
        pyautogui.moveTo(event.x, event.y)

    @staticmethod
    def click(event: Click):
        pass

    @staticmethod
    def double_click(event: DoubleClick):
        pass


def main():
    capture = cv.VideoCapture(0)
    vm = VirtualMouse(capture, HandDetector(max_hands=1), 1280, 720, DEBUG)
    previous_frame_time = time()
    EventDispatcher.add_listener(MouseListener())

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
