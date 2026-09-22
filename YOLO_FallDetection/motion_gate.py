"""
motion_gate.py
프레임 차분 기반 움직임 감지 유틸리티.

낙상 감지 파이프라인(run_camera.py)에서 "움직임이 없는 평상시 프레임"에는
무거운 YOLO(tflite) 추론을 건너뛰기 위한 전처리 단계로 사용합니다.

주의: 낙상이 확정된 이후(state_machine.py의 FallStateMachine이 "fallen"
상태일 때)는 이 게이팅을 적용하지 않고 계속 추론해야 합니다.
쓰러진 뒤 움직이지 않는 상태 자체가 감지해야 할 위험 신호이기 때문입니다.
이 조건 분기는 run_camera.py의 메인 루프에서 처리합니다.
"""

import cv2


class MotionDetector:
    def __init__(
        self,
        motion_threshold_area: int = 1500,  # 이 면적(px^2)보다 큰 변화가 있어야 "움직임"으로 판단.
                                             # 카메라 해상도(640x480 기준)/설치 거리에 맞게 튜닝 필요.
        diff_threshold: int = 25,           # 프레임 간 밝기 차이 임계값 (0~255). 낮을수록 민감.
        blur_ksize: int = 21,               # 노이즈 제거용 가우시안 블러 커널 크기(홀수)
    ):
        self.motion_threshold_area = motion_threshold_area
        self.diff_threshold = diff_threshold
        self.blur_ksize = blur_ksize
        self._prev_gray = None

    def detect(self, frame) -> bool:
        """이번 프레임에 의미 있는 움직임이 있으면 True, 없으면 False."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False  # 첫 프레임은 비교 대상이 없어 "움직임 없음"으로 처리

        frame_delta = cv2.absdiff(self._prev_gray, gray)
        thresh = cv2.threshold(frame_delta, self.diff_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = max((cv2.contourArea(c) for c in contours), default=0)
        self._prev_gray = gray

        return max_area > self.motion_threshold_area
