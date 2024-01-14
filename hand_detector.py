from __future__ import annotations
from functools import reduce
from collections import namedtuple
import mediapipe
import numpy as np
import cv2 as cv
from typing import NamedTuple, Optional, List
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from debug import Debugger

Fingers = namedtuple("Fingers", ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY"))(4, 8, 12, 16, 20)


class Finger:
    def __init__(self, landmarks=None, tip: int = -1):
        self.landmarks = landmarks if landmarks is not None else []
        self.tip: int = tip

    def up(self, left: bool = False) -> bool:
        if len(self.landmarks) == 0:
            return False

        if self.tip == Fingers.THUMB:
            if left:
                up = self.landmarks[-1][1] < self.landmarks[-2][1]
            else:
                up = self.landmarks[-1][1] > self.landmarks[-2][1]
        else:
            up = self.landmarks[-1][2] < self.landmarks[-3][2]

        return up


class Hand:
    def __init__(self, image: np.ndarray, left: bool, wrist=None, landmarks: Optional[List[int]] = None,
                 fingers: Optional[List[Finger]] = None):
        self.image = image
        self.left = left
        self.wrist = wrist
        self.fingers = fingers if fingers is not None else []
        self.landmarks = landmarks if landmarks is not None else []

    def get_finger_by_tip(self, tip: int) -> Optional[Finger]:
        return next((finger for finger in self.fingers if finger.tip == tip), None)

    @property
    def landmarks(self) -> List:
        landmarks = [self.wrist]

        for finger in self.fingers:
            landmarks.extend(landmark for landmark in finger.landmarks)

        return landmarks

    @landmarks.setter
    def landmarks(self, landmarks: list):
        h, w, _ = self.image.shape
        self.fingers = [Finger(tip=i * 4) for i in range(1, 6)]

        for landmark_number, landmark in enumerate(landmarks):
            if landmark_number == 0:
                self.wrist = [landmark_number, int(landmark.x * w), int(landmark.y * h)]
                continue

            for finger in self.fingers:
                if landmark_number <= finger.tip:
                    finger.landmarks.append([landmark_number, int(landmark.x * w), int(landmark.y * h)])
                    break

    def fingers_up(self) -> List[bool]:
        return [finger.up(self.left) for finger in self.fingers]

    def find_distance(self, landmark1: int, landmark2: int) -> int:
        def diff(first, second): return second - first

        return np.hypot(reduce(diff, self.landmarks[landmark1][1:3]), reduce(diff, self.landmarks[landmark2][1:3]))


class HandDetector:
    def __init__(
            self,
            debug_mode: bool = False,
            mode=False,
            max_hands: int = 2,
            detection_confidence: float = 0.5,
            track_confidence: float = 0.5
    ):
        self.image: Optional[np.ndarray] = None
        self.results: Optional[NamedTuple] = None

        self.debug_mode = debug_mode
        self.mode = mode

        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.hands_pipe = mediapipe.solutions.hands
        self.draw_utils = mediapipe.solutions.drawing_utils

        self.hands_pipe = self.hands_pipe.Hands(
            self.mode, self.max_hands, 0, self.detection_confidence, self.track_confidence
        )

    @Debugger
    def debug(self):
        if not self.debug_mode or self.results.multi_hand_landmarks is None:
            return

        for hand_landmarks in self.results.multi_hand_landmarks:
            self.draw_utils.draw_landmarks(self.image, hand_landmarks, HAND_CONNECTIONS)

        for hand in self.hands:
            xx, yy = tuple(landmark[1] for landmark in hand.landmarks), tuple(landmark[2] for landmark in hand.landmarks)

            for finger in hand.fingers:
                if finger.up(hand.left):
                    cv.circle(self.image, finger.landmarks[-1][1:3], 15, (0, 0, 255), cv.FILLED)

            for index, finger in enumerate(hand.fingers[1:]):
                x1, y1 = finger.landmarks[-1][1:3]
                x2, y2 = hand.fingers[index].landmarks[-1][1:3]
                cv.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv.circle(self.image, (int(x1 + x2) // 2, int(y1 + y2) // 2), 15, (0, 255, 0), cv.FILLED)

            min_x, max_x = min(xx), max(xx)
            min_y, max_y = min(yy), max(yy)
            cv.rectangle(self.image, (min_x - 20, min_y - 20), (max_x + 20, max_y + 20), (255, 255, 255), 2)

    @property
    @debug.after(recursive=False)
    def hands(self):
        image_rgb = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        self.results = self.hands_pipe.process(image_rgb)

        if self.results.multi_hand_landmarks is not None:
            return [
                Hand(self.image, classification.classification[0].index == 1,
                     landmarks=self.results.multi_hand_landmarks[hand_number].landmark)
                for hand_number, classification in enumerate(self.results.multi_handedness)
            ]

        return []
