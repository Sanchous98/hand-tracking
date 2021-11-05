import cv2.cv2 as cv
from hand_detector import HandDetector, Fingers
import autopy
import numpy as np


class VirtualMouse:
    def __init__(self, capture: cv.VideoCapture, hand_detector: HandDetector, camera_width: int, camera_height: int,
                 debug_mode: bool = False):
        self.hand_detector = hand_detector
        self.hand_detector.debug_mode = debug_mode
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.capture = capture
        self.debug_mode = debug_mode

        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, camera_width)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, camera_height)

        self.screen_width, self.screen_height = autopy.screen.size()
        self.screen_width, self.screen_height = int(self.screen_width), int(self.screen_height)
        scale = self.screen_height / self.camera_height / 0.75
        self.box_width, self.box_height = self.screen_width / scale, self.screen_height / scale

    def debug(self):
        x1, y1 = (int((self.camera_width - self.box_width) / 2), int((self.camera_height - self.box_height) / 2))
        x2, y2 = self.camera_width - x1, self.camera_height - y1
        cv.rectangle(self.hand_detector.image, (x1, y1), (x2, y2), (255, 255, 255), 3)

    def get_image(self):
        box_x1, box_y1 = int((self.camera_width - self.box_width) / 2), int((self.camera_height - self.box_height) / 2)
        box_x2, box_y2 = self.camera_width - box_x1, self.camera_height - box_y1

        _, image = self.capture.read()
        self.hand_detector.image = image
        hands = self.hand_detector.find_hands()

        for hand in hands:
            x1, y1 = hand.get_finger_by_tip(Fingers.INDEX).landmarks[0][1:3]
            # x2, y2 = hand.get_finger_by_tip(Finger.TIPS.MIDDLE).landmarks[0][1:3]

            ups = hand.fingers_up()

            if ups[1] or ups[1] and ups[2] or ups[1] and ups[2] and ups[3]:
                x3 = np.interp(x1, (box_x1, box_x2), (0, self.screen_width))
                y3 = np.interp(y1, (box_y1, box_y2), (0, self.screen_height))

                try:
                    autopy.mouse.move(self.screen_width - x3, y3)
                except ValueError:
                    if x3 <= 0:
                        x3 = 1

                    if y3 <= 0:
                        y3 = 1

                    if x3 >= self.screen_width:
                        x3 = self.screen_width - 1

                    if y3 >= self.screen_height:
                        y3 = self.screen_height - 1

                    autopy.mouse.move(self.screen_width - x3, y3)
            else:
                pass
                # length, image, _ = self.hand_detector.find_distance(self.hand_detector.fingers.index,
                #                                                     self.hand_detector.fingers.middle, image)

        if self.debug_mode:
            self.debug()

        return image
