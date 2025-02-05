import sys
import streamlit as st
import torch
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2
import pathlib
from PIL import Image
from pathlib import Path

# 🔹 Windows 환경에서 'PosixPath' 대신 'WindowsPath'를 사용하도록 설정
if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 경로 추가
YOLOV5_PATH = str("./")
sys.path.append(YOLOV5_PATH)

# YOLOv5에서 필요한 모듈 불러오기
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox  # YOLOv5 리사이징 방식 사용

# YOLOv5 모델 로드 함수
@st.cache_resource(hash_funcs={DetectMultiBackend: lambda _: None})
def load_model():
    model_path = str("./runs/train/fold_2/weights/best.pt")
    device = select_device('cpu')
    model = DetectMultiBackend(model_path, device=device, dnn=False)
    return model

# 🔹 이미지 전처리 함수 (외곽선 검출 후 크롭 & YOLOv5 방식 리사이즈)
def preprocess_image(image, final_size=(500, 500), margin=20):
    """
    1. 이미지를 그레이스케일로 변환 후 이진화
    2. 외곽선을 검출하여 가장 큰 영역을 찾음
    3. 해당 영역을 크롭 후 지정된 크기로 리사이즈
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        height, width, _ = image.shape
        x_new = max(0, x - margin)
        y_new = max(0, y - margin)
        x_end = min(width, x + w + margin)
        y_end = min(height, y + h + margin)

        cropped = image[y_new:y_end, x_new:x_end]

        # 리사이즈 (비율 유지 없이 강제 변환)
        resized_image = cv2.resize(cropped, final_size, interpolation=cv2.INTER_LINEAR)
        return resized_image
    else:
        return cv2.resize(image, final_size, interpolation=cv2.INTER_LINEAR)

# Streamlit UI
st.title("Custom YOLOv5 Object Detection (Local Model)")
st.write("내 컴퓨터의 YOLOv5 모델을 사용하여 객체를 감지합니다.")

# 모델 불러오기
model = load_model()

# YOLOv5 클래스 네임 강제 설정
custom_names = ['extruded', 'crack', 'cutting', 'side stamped', 'object']
model.names = custom_names
names = model.names

st.write("Model Class Names:", names)

# 이미지 업로드
uploaded_files = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# 세션 상태에서 현재 이미지 인덱스 저장 (초기값: 0)
if "image_index" not in st.session_state:
    st.session_state.image_index = 0

if uploaded_files:
    # ✅ 한 개의 파일만 선택되었을 경우 리스트로 변환
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    total_images = len(uploaded_files)

    # 🎯 **드래그 바(Slider) 추가 (이미지가 1개 이상일 때만 표시)**
    if total_images > 1:
        selected_index = st.slider("이미지 선택", 0, total_images - 1, st.session_state.image_index)
        if selected_index != st.session_state.image_index:
            st.session_state.image_index = selected_index
            st.rerun()

    current_index = st.session_state.image_index
    uploaded_file = uploaded_files[current_index]
    image = Image.open(uploaded_file)
    image = np.array(image)

    # (⭐ 추가) 이미지 전처리 적용
    processed_image = preprocess_image(image)

    # YOLOv5 리사이징 방식 적용 (비율 유지 + 패딩)
    img_resized, ratio, pad = letterbox(processed_image, new_shape=640, auto=True)

    # YOLOv5 모델 입력 처리
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # HWC → CHW
    img_resized = np.ascontiguousarray(img_resized)

    img_tensor = torch.from_numpy(img_resized).to(model.device)
    img_tensor = img_tensor.float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = non_max_suppression(pred, 0.25, 0.45)

    detected_image = processed_image.copy()

    for det in pred:
        if det is not None and len(det):
            # ⚠️ YOLOv5 방식으로 원본 크기에 맞게 바운딩 박스 변환
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], processed_image.shape).round()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)

                class_name = names[class_id] if class_id < len(names) else f"Class {class_id}"
                label = f"{class_name} {conf:.2f}"

                cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(detected_image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(detected_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    st.image(detected_image, caption=f"Detection Results - {uploaded_file.name}", use_column_width=True)
    st.write(f"객체 감지 완료 - {uploaded_file.name}")

    # **"이전" 및 "다음" 버튼 (이미지가 2개 이상일 때만 표시)**
    if total_images > 1:
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.session_state.image_index > 0:
                if st.button("이전 이미지"):
                    st.session_state.image_index -= 1
                    st.rerun()

        with col2:
            if st.session_state.image_index < total_images - 1:
                if st.button("다음 이미지"):
                    st.session_state.image_index += 1
                    st.rerun()
