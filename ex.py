import cv2
import streamlit as st
from mediapipe import solutions
import mediapipe as mp
import numpy as np

# Streamlit 페이지 설정
st.title("Blink Detection with OpenCV and MediaPipe")

# MediaPipe 얼굴 메쉬 모듈 불러오기
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# 눈 비율(EAR)을 계산하는 함수
def calculate_EAR(eye_landmarks, landmarks, image_shape):
    h, w, _ = image_shape
    p1 = np.array([landmarks[eye_landmarks[0]].x * w, landmarks[eye_landmarks[0]].y * h])
    p2 = np.array([landmarks[eye_landmarks[1]].x * w, landmarks[eye_landmarks[1]].y * h])
    p3 = np.array([landmarks[eye_landmarks[2]].x * w, landmarks[eye_landmarks[2]].y * h])
    p4 = np.array([landmarks[eye_landmarks[3]].x * w, landmarks[eye_landmarks[3]].y * h])
    p5 = np.array([landmarks[eye_landmarks[4]].x * w, landmarks[eye_landmarks[4]].y * h])
    p6 = np.array([landmarks[eye_landmarks[5]].x * w, landmarks[eye_landmarks[5]].y * h])

    # EAR 계산
    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear

# EAR 임계값 및 연속 프레임 깜박임 기준
EAR_THRESHOLD = 0.2

# Streamlit 카메라 입력을 사용하여 이미지 캡처
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # 이미지를 OpenCV 형식으로 변환
    bytes_data = img_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # BGR을 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe로 얼굴 메쉬 추적
    result = face_mesh.process(frame_rgb)

    # 얼굴이 감지되면 랜드마크 그리기 (눈 주위만)
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # 눈 랜드마크 그리기
            left_eye_landmarks = [33, 160, 158, 133, 153, 144]  # 왼쪽 눈
            right_eye_landmarks = [362, 385, 387, 263, 373, 380]  # 오른쪽 눈

            # 왼쪽 및 오른쪽 눈의 EAR 계산
            left_EAR = calculate_EAR(left_eye_landmarks, face_landmarks.landmark, frame.shape)
            right_EAR = calculate_EAR(right_eye_landmarks, face_landmarks.landmark, frame.shape)

            # 양쪽 눈의 평균 EAR
            ear = (left_EAR + right_EAR) / 2.0

            # EAR 값 화면에 표시
            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 경고 문구 화면에 표시
            if ear < EAR_THRESHOLD:
                cv2.putText(frame, 'WARNING: Eyes closed!', (frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Streamlit에서 프레임 표시
    st.image(frame, channels="BGR")
