# 모듈 로딩
import tflite_runtime.interpreter as tflite
import numpy as np
import time
import cv2

# LiteRT 모델 선택
modelPath = "best.tflite"
#modelPath = "best_int8_project.tflite"
#modelPath = "best_w8a32_project.tflite"
print('model path:', modelPath)

# LiteRT 모델 로딩
interpreter = tflite.Interpreter(model_path = modelPath) # 모델 로딩
interpreter.allocate_tensors() # tensor 할당

# 모델 정보 얻기 
input_details = interpreter.get_input_details()  # input tensor 정보 얻기
output_details = interpreter.get_output_details() # output tensor 정보 얻기
print(input_details)
print(output_details)
input_index = input_details[0]['index']
output_index = output_details[0]['index']
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']
height = input_details[0]['shape'][2]
width = input_details[0]['shape'][3]
print('model input shape:', (height, width))

# BB 텍스트 및 색상 정의
ansToText = {0:'scissors', 1:'rock', 2:'paper'}
colorList = [(255,0,0),(0,255,0),(0,0,255)]

# 모델 입력 크기
IMG_SIZE = 320

# Threshold 설정
CONF_TH = 0.4
IOU_TH  = 0.45

def letterbox(img, new_shape=(320,320), color=(114,114,114)):
    h, w = img.shape[:2]    
    nh, nw = new_shape
    r = min(nw / w, nh / h)

    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))

    pad_w = nw - new_w
    pad_h = nh - new_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_y, pad_y,
        pad_x, pad_x,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, r, pad_x, pad_y

def processImage(frame):

    # BGR을 RGB로 변경
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # letterbox 적용
    img_lb, r, pad_x, pad_y = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))

    # 0 ~ 1 사이 값으로 변경
    img = img_lb.astype(np.float32) / 255.0

    # 모델의 입력 형태로 수정: (1,3,320,320)
    #   최상위 차원 증가: (320,320,3) -> (1,320,320,3)
    img = np.expand_dims(img, axis=0)
    #   축 위치 변경: (1,320,320,3) -> (1,3,320,320)
    img = np.transpose(img, (0, 3, 1, 2))

    # 모델에 입력하여 결과 얻기
    #   input tensor 설정
    interpreter.set_tensor(input_index, img)
    #   모델 실행
    interpreter.invoke()
    #   output tensor 얻기: (1,7,2100) -> (7,2100) -> (2100,7)
    raw = interpreter.get_tensor(output_index)[0].transpose()

    # raw에서 각 Object 별로 confidence score의 최대값과 해당 클래스의 id를 추출
    class_scores = raw[:, 4:]                      # confidence score: (2100, 3)
    confidences = np.max(class_scores, axis=1)     # confidence score의 최대값: (2100,)
    class_ids = np.argmax(class_scores, axis=1)    # 해당 클래스의 id: (2100,)

    # confidence score가 설정한 임계값(CONF_TH)보다 높은 Object만 필터링
    keep_mask = confidences > CONF_TH
    filtered_raw = raw[keep_mask] # (N,7)
    scores = confidences[keep_mask] # (N,)
    classes = class_ids[keep_mask]  # (N,)

    # 중심점 좌표 [cx, cy, w, h]를 추출하고 좌상단 좌표 [x, y, w, h]로 변환
    cx, cy, w, h = filtered_raw[:, 0], filtered_raw[:, 1], filtered_raw[:, 2], filtered_raw[:, 3]
    x = cx - (w / 2)
    y = cy - (h / 2)
    boxes = np.stack([x, y, w, h], axis=-1) # (N,4)

    # Batched NMS 처리
    keep = cv2.dnn.NMSBoxesBatched(
            boxes, 
            scores, 
            classes, 
            score_threshold=CONF_TH, 
            nms_threshold=IOU_TH
    )
    
    # 최종 BB만 그리기
    draw_boxes = [(boxes[i], scores[i], classes[i]) for i in keep]

    boxes_pixel = []
    for (x,y,w,h), sc, cid in draw_boxes:
        x *= IMG_SIZE; y *= IMG_SIZE
        w *= IMG_SIZE; h *= IMG_SIZE

        x1 = (x - pad_x) / r
        y1 = (y - pad_y) / r
        x2 = (x + w - pad_x) / r
        y2 = (y + h - pad_y) / r

        x1 = int(np.clip(x1,0,frame.shape[1]))
        y1 = int(np.clip(y1,0,frame.shape[0]))
        x2 = int(np.clip(x2,0,frame.shape[1]))
        y2 = int(np.clip(y2,0,frame.shape[0]))

        # 프로젝트 1. 좌/우 플레이어 구분
        # 좌표를 비교해서 P1(왼쪽), P2(오른쪽)
        boxes_pixel.append([x1,y1,x2,y2,sc, cid])

    boxes_pixel.sort(key= lambda x :x[0])

    for i , (x1,y1,x2,y2,sc,cid) in enumerate(boxes_pixel):
        player = (1 if i == 0 else 2) if len(boxes_pixel) == 2 else 1

        # BB 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[cid], 2)

        # 판정 결과 표시
        cv2.putText(frame, f"Player : {player}", (x1,y1-24), cv2.FONT_HERSHEY_PLAIN, 1, colorList[cid], 2)
        cv2.putText(frame, f'{ansToText[cid]} {int(sc*100)}%', (x1,y1-7), cv2.FONT_HERSHEY_PLAIN, 1, colorList[cid], 2)

    # 프로젝트3 : 손이 2개가 아닐 때 예외처리
    if len(keep) != 2:
        round_result = '2 HANDS ARE NEEDED'
    else:
        # 프로젝트2 : 승자 찾기
        # {0:'scissors', 1:'rock', 2:'paper'}
        p1 = boxes_pixel[0][5]  # class id
        p2 = boxes_pixel[1][5]

        if p1 == p2:
            round_result = 'DRAW'
        elif p2 == ((p1+2) % 3) :
            round_result = 'P1 WIN'
        else:
            round_result = 'P2 WIN'

    return round_result

# 스코어보드 / 패딩 설정
PAD_HEIGHT = 70
score = {'P1 WIN': 0, 'P2 WIN': 0, 'DRAW': 0}

# 카메라 설정
cap = cv2.VideoCapture(0) # 0번 카메라 열기
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

# 윈도우 설정
cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cam', 320+40, 240+60+PAD_HEIGHT)

startTime = time.time()

# 게임 상태 관리
state = 'LIVE'
countdown_start = None
frozen_frame = None
frozen_result = None

while(cap.isOpened()):
    if state == 'FROZEN':
        frame = frozen_frame.copy()
        round_result = frozen_result
    else:
        ret,frame=cap.read() # 사진 찍기 -> (240,320,3)
        if not ret: break

        # 이미지 처리
        round_result = processImage(frame)

    # FPS 표시
    curTime = time.time()
    fps = 1/(curTime - startTime)
    startTime = curTime
    cv2.putText(frame,f'FPS: {fps:.1f}',(20, 50),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),2)

    if state == 'COUNTDOWN':
        elapsed = time.time() - countdown_start
        time_remaining = 3 - int(elapsed)

        if time_remaining > 0:
            cv2.putText(frame, str(time_remaining), (140, 140), cv2.FONT_HERSHEY_PLAIN, 6, (0,255,255), 4)
        else:
            # 카운트다운 종료 -> 이 순간의 frame(판정 결과 포함)을 고정
            frozen_frame = frame.copy()
            state = 'FROZEN'
            frozen_result = round_result
            if frozen_result in score:
                score[frozen_result] += 1

    # ---- 하단 패딩 영역 추가 ----
    canvas = cv2.copyMakeBorder(frame, 0, PAD_HEIGHT, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    cv2.putText(canvas, round_result, (10, frame.shape[0] + 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

    scoreboard_text = f"P1 {score['P1 WIN']}  :  {score['P2 WIN']} P2   (DRAW {score['DRAW']})"
    cv2.putText(canvas, scoreboard_text, (10, frame.shape[0] + 55), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    # 이미지 출력
    cv2.imshow('cam',canvas)

     # 10ms 동안 키 입력 대기
    key = cv2.waitKey(10)
    if  key == ord('q'): break
    # 프로젝트 4 : 3,2,1 카운트 다운 후 판정 순간 프레임 고정 후 결과 보여주기, space 누르면 해제
    elif key == ord('r') and state == 'LIVE':
        state = 'COUNTDOWN'
        countdown_start = time.time()

    # 프로젝트 5 : 재시작 키 추가   
    elif key == ord(' ') and state == 'FROZEN':
        state = 'LIVE'
        frozen_frame = None



cap.release() # 카메라 닫기
cv2.destroyAllWindows() # 모든 창 닫기
