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

# ---- 설정 -----------------------------------------------------------------
MODEL_PATH = "best_int8.tflite"   # 다른 변환본과 비교하려면 이 줄만 바꾸면 됨
IMG_SIZE = 320                     # Colab 학습 때(config.IMGSZ)와 반드시 동일해야 함
CONF_TH = 0.3                       # 신뢰도 기준 
IOU_TH = 0.45                       # NMS 기준
CONFIRM_SEC = 1.0                  # 상태 기계: fall이 이 시간(초) 이상 지속돼야 확정
INFO_PAD_HEIGHT = 60               # 하단 정보 표시용 검은 여백 높이(px)
NUM_THREADS = 4                    # 라즈베리파이 코어 수에 맞춰 조정
DEBUG = False                      # 진단용 로그 on/off

CLASS_NAMES = {0: "fall", 1: "non-fall"}   # ← data.yaml 순서로 반드시 확인/수정
FALL_ID = [k for k, v in CLASS_NAMES.items() if v == "fall"][0]
BOX_COLOR = {"fall": (0, 0, 255), "non-fall": (0, 200, 0)}


def letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE), color=(114, 114, 114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)
    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))
    pad_w, pad_h = nw - new_w, nh - new_h
    pad_x, pad_y = pad_w // 2, pad_h // 2
    padded = cv2.copyMakeBorder(resized, pad_y, pad_y, pad_x, pad_x,
                                 cv2.BORDER_CONSTANT, value=color)
    return padded, r, pad_x, pad_y


def main():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("input :", input_details)
    print("output:", output_details)
    print(f"클래스: {CLASS_NAMES}  (fall = {FALL_ID})  ← data.yaml과 순서가 같은지 확인하세요")
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    sm = FallStateMachine(confirm_sec=CONFIRM_SEC)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다. `ls /dev/video*` 로 장치를 확인하세요.")

    prev_t = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # ---- 전처리 (YOLO_Project.py의 processImage와 동일) --------------
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_lb, r, pad_x, pad_y = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))
        img = img_lb.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))          # (1,H,W,3) -> (1,3,H,W)

        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_index)[0].transpose()   # (N, 4+nc)

        # ---- 후처리: 신뢰도 필터 + NMS -----------------------------------
        class_scores = raw[:, 4:]
        confidences = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        keep_mask = confidences > CONF_TH
        filtered = raw[keep_mask]
        scores = confidences[keep_mask]
        classes = class_ids[keep_mask]

        cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=-1)   # 정규화 좌표(0~1)

        detections = []
        detection_scores = []
        if len(boxes) > 0:
            keep = cv2.dnn.NMSBoxesBatched(boxes, scores, classes,
                                            score_threshold=CONF_TH, nms_threshold=IOU_TH)
            if DEBUG and len(keep) > 0:
                print(f"[DEBUG] 이번 프레임 검출 수: {len(keep)}")
            for i in keep:
                x, y, bw, bh = boxes[i] * IMG_SIZE
                x1 = int(np.clip((x - pad_x) / r, 0, frame.shape[1]))
                y1 = int(np.clip((y - pad_y) / r, 0, frame.shape[0]))
                x2 = int(np.clip((x + bw - pad_x) / r, 0, frame.shape[1]))
                y2 = int(np.clip((y + bh - pad_y) / r, 0, frame.shape[0]))
                label_i = CLASS_NAMES[int(classes[i])]
                detections.append(label_i)
                detection_scores.append(scores[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR[label_i], 2)
                cv2.putText(frame, f"{label_i} {scores[i]*100:.0f}%", (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, BOX_COLOR[label_i], 2)
                if DEBUG:
                    print(f"[DEBUG] label={label_i} score={scores[i]:.3f}")

        # 프레임 내 한 명이라도 fall이면 fall로 집계
        if "fall" in detections:
            label = "fall"
            frame_score = max(s for lbl, s in zip(detections, detection_scores) if lbl == "fall")
        elif detections:
            label = "non-fall"
            frame_score = max(detection_scores)
        else:
            label = "none"
            frame_score = 0.0

        # ---- 상태 기계 ----------------------------------------------------
        now = time.time()
        alarm = sm.update(label, now)
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        if alarm:
            print(f"[ALARM] 낙상 확정  t={now:.1f}s")
            # TODO: 여기에 GPIO 부저/LED 트리거 추가

        # ---- 표시용 캔버스: 프레임 아래에 검은 여백을 붙여서 정보 표시 ----
        h, w = frame.shape[:2]
        canvas = np.zeros((h + INFO_PAD_HEIGHT, w, 3), dtype=np.uint8)
        canvas[:h, :] = frame

        hud_color = (0, 0, 255) if sm.state == "fallen" else (0, 255, 255)
        cv2.putText(canvas, f"state:{sm.state} score:{frame_score*100:.0f}%", (10, h + 22),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, hud_color, 2)
        cv2.putText(canvas, f"fps:{fps:.1f}", (10, h + 45),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, hud_color, 2)

        cv2.imshow("fall detection", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()