"""
라즈베리파이에서 실행 — 가위바위보 프로젝트(YOLO_Project.py)와 동일한 tflite_runtime 방식.
전처리(letterbox)·후처리(박스 디코딩, NMS) 로직은 그 코드를 그대로 재사용했습니다.
ultralytics 설치가 필요 없습니다 — 이미 검증된 환경(tflite_runtime + opencv + numpy) 그대로 씁니다.

사전 준비
  1) 이 파일과 같은 폴더에 tflite 파일을 둘 것 (예: best_int8.tflite)
  2) 아래 CLASS_NAMES를 Colab data.yaml의 names 순서와 반드시 맞출 것
     (확인법: Colab에서 `print(yaml.safe_load(open('data.yaml'))['names'])`)
     순서가 다르면 fall과 non-fall이 뒤바뀐 채로 동작합니다.

실행: python3 run_camera.py
종료: 영상 창에서 q
"""
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from state_machine import FallStateMachine

# ---- 설정 -------------------------------------------------
MODEL_PATH = "best_int8.tflite" 
IMG_SIZE = 320                    
CONF_TH = 0.45                    # 신뢰도 임계값
IOU_TH = 0.45
CONFIRM_SEC = 1.0                 # 상태 기계: 낙상 지속 시간 (초)
MOTION_TH = 3.0                   # 정적 사물/배경 필터링 임계값
IMPACT_TH = 18.0                  # 천천히 눕기 구분을 위한 '급격한 낙상' 움직임 임계값
INFO_PAD_HEIGHT = 60              # 하단 정보 표시용 검은 여백 높이(px)

CLASS_NAMES = {0: "fall", 1: "non-fall"} 
FALL_ID = [k for k, v in CLASS_NAMES.items() if v == "fall"][0]
BOX_COLOR = {
    "fall": (0, 0, 255),       # 빨강 (낙상 확정/진행)
    "non-fall": (0, 255, 0),   # 초록 (정상)
    "static": (128, 128, 128)  # 회색 (가만히 있는 물체/배경)
}

def draw_hud(canvas, h, text_lines, color):
    for i, line in enumerate(text_lines):
        cv2.putText(canvas, line, (10, h + 22 + i * 23),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

def main():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]
    input_type = input_details[0]["dtype"]
    input_scale_zero = input_details[0]["quantization"]

    sm = FallStateMachine(confirm_sec=CONFIRM_SEC)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다.")

    prev_t = time.time()
    prev_gray = None

    fall_hold_frames = 0
    MAX_FALL_HOLD = 20  # 쓰러진 후 인식이 잠시 끊겨도 상태 유지하는 프레임 수

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_h, frame_w, _ = frame.shape

        # ---- 1. 정교한 움직임 감지 (이진 마스크 생성) ----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        frame_diff = cv2.absdiff(prev_gray, gray)
        prev_gray = gray.copy()

        # 미세한 노이즈를 날리고 확실히 움직인 픽셀만 흰색(255)으로 추출하는 임계값 처리
        _, motion_mask = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)

        # ---- 모델 전처리 ----
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        
        if input_type == np.int8 or input_type == np.uint8:
            scale, zero_point = input_scale_zero
            if scale != 0:
                img_norm = img_resized.astype(np.float32) / scale + zero_point
            else:
                img_norm = img_resized.astype(np.float32)
            img = img_norm.astype(input_type)
        else:
            img = img_resized.astype(np.float32) / 255.0

        img = np.expand_dims(img, axis=0)
        if img.shape[-1] == 3:
            img = np.transpose(img, (0, 3, 1, 2))

        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        
        raw = interpreter.get_tensor(output_index)
        if raw.ndim == 3:
            raw = raw[0]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.transpose()

        # ---- 후처리: 신뢰도 필터 + NMS ----
        class_scores = raw[:, 4:]
        confidences = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        keep_mask = confidences > CONF_TH
        filtered = raw[keep_mask]
        scores = confidences[keep_mask]
        classes = class_ids[keep_mask]

        if len(filtered) > 0:
            cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
            boxes = np.stack([
                (cx - w / 2) * IMG_SIZE, 
                (cy - h / 2) * IMG_SIZE, 
                w * IMG_SIZE, 
                h * IMG_SIZE
            ], axis=-1)
        else:
            boxes = np.array([])

        label = "non-fall"
        found_fall = False

        if len(boxes) > 0:
            keep = cv2.dnn.NMSBoxesBatched(boxes.tolist(), scores.tolist(), classes.tolist(),
                                         score_threshold=CONF_TH, nms_threshold=IOU_TH)
            if len(keep) > 0:
                scale_x = frame_w / IMG_SIZE
                scale_y = frame_h / IMG_SIZE

                for i in keep:
                    x, y, bw, bh = boxes[i]
                    
                    x1 = int(np.clip(x * scale_x, 0, frame_w))
                    y1 = int(np.clip(y * scale_y, 0, frame_h))
                    x2 = int(np.clip((x + bw) * scale_x, 0, frame_w))
                    y2 = int(np.clip((y + bh) * scale_y, 0, frame_h))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    # ---- [필터 1] 바닥에 깔린 납작한 사물(매트리스 등) 오인식 원천 차단 ----
                    if box_h > 0 and (box_w / box_h > 2.3) and (y2 > frame_h * 0.45):
                        color = BOX_COLOR["static"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(frame, "flat object", (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)
                        continue

                    # ---- [필터 2] 박스 내부에 실제 움직이는 픽셀 비율 검사 (가만히 있는 사물 완벽 차단) ----
                    box_diff = frame_diff[y1:y2, x1:x2]
                    box_mask = motion_mask[y1:y2, x1:x2]
                    
                    mean_motion = np.mean(box_diff) if box_diff.size > 0 else 0
                    # 박스 넓이 대비 움직인 픽셀(흰색)의 비율 계산 (0.0 ~ 1.0)
                    active_ratio = np.count_nonzero(box_mask) / box_mask.size if box_mask.size > 0 else 0

                    current_class_id = int(classes[i])
                    current_label = CLASS_NAMES[current_class_id]

                    is_fall_in_progress = (sm.state != "normal")

                    # 만약 사람이 서 있거나 평소 상태(non-fall)인데 움직임이 거의 없다면 -> 초록색 non-fall 유지
                    # 만약 모델이 fall을 예측했는데, 움직이는 픽셀 비율이 너무 낮고(예: 3% 미만) 쓰러지는 중이 아니라면 -> 가만히 있는 사물/배경으로 취급(static)
                    if current_label == "fall":
                        if active_ratio < 0.03 and mean_motion < MOTION_TH:
                            if is_fall_in_progress:
                                found_fall = True  # 이미 쓰러져서 누워있는 상태라면 허용
                            else:
                                color = BOX_COLOR["static"]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                                cv2.putText(frame, f"static {scores[i]*100:.0f}%", (x1, max(y1 - 8, 12)),
                                            cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)
                                continue

                        # 천천히 눕거나 앉는 동작 구별 (변화량이 부족하면 non-fall로 강등)
                        if mean_motion < IMPACT_TH and not is_fall_in_progress:
                            current_label = "non-fall"
                        else:
                            found_fall = True

                    # 시각화 출력
                    color = BOX_COLOR.get(current_label, (255, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{current_label} {scores[i]*100:.0f}%", (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_PLAIN, 1.2, color, 2)

                if found_fall:
                    label = "fall"

        # ---- 쓰러진 후 바닥에 누워있을 때 상태 유지 유예 로직 ----
        is_fall_state = (sm.state != "normal")
        if found_fall:
            fall_hold_frames = MAX_FALL_HOLD
        elif is_fall_state and fall_hold_frames > 0:
            label = "fall"
            fall_hold_frames -= 1

        # ---- 상태 기계 업데이트 ----
        now = time.time()
        alarm = sm.update(label, now)
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        # 표시용 캔버스 생성 (하단 여백 추가)
        canvas = np.zeros((frame_h + INFO_PAD_HEIGHT, frame_w, 3), dtype=np.uint8)
        canvas[:frame_h, :] = frame

        hud_color = (0, 0, 255) if sm.state == "fallen" else (0, 255, 255)
        draw_hud(
            canvas, frame_h,
            [f"state:{sm.state}  fps:{fps:.1f}"],
            hud_color,
        )

        if alarm:
            print(f"[ALARM] 낙상 확정  t={now:.1f}s")

        cv2.imshow("fall detection", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()