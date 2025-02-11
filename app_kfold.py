import sys
import streamlit as st
import torch
import numpy as np
import os
import cv2
import pathlib
from PIL import Image
from pathlib import Path
import torchvision.ops as ops
import matplotlib.pyplot as plt

# Windows 환경에서 'PosixPath' 대신 'WindowsPath'를 사용하도록 설정
if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 경로 추가
YOLOV5_PATH = str("./")
sys.path.append(YOLOV5_PATH)

# YOLOv5에서 필요한 모듈 불러오기
from models.experimental import attempt_load
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# ✅ 사용할 YOLOv5 모델 5개 설정 (K-Fold 모델)
MODEL_PATHS = [
    "./runs/train/fold_1/weights/best_fold1.pt",
    "./runs/train/fold_2/weights/best_fold2.pt",
    "./runs/train/fold_3/weights/best_fold3.pt",
    "./runs/train/fold_4/weights/best_fold4.pt",
    "./runs/train/fold_5/weights/best_fold5.pt",
]

# ✅ 개별 YOLOv5 모델 추가
SINGLE_MODEL_PATH = "./runs/train/exp/weights/best.pt"

# ✅ YOLOv5 모델 5개 로드 (K-Fold)
@st.cache_resource
def load_models():
    return [attempt_load(path) for path in MODEL_PATHS]

# ✅ 개별 YOLOv5 모델 로드
@st.cache_resource
def load_single_model():
    device = select_device('cpu')
    return DetectMultiBackend(SINGLE_MODEL_PATH, device=device, dnn=False)

# ✅ UI 개선: 제목 및 모델 선택 인터페이스 향상
st.markdown("<h1 style='text-align: center; font-size: 50px;'>O-RING 불량검출</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 30px;'>사용할 모델을 선택하세요</h2>", unsafe_allow_html=True)

# 기본적으로 K-Fold 앙상블이 선택되도록 설정
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "K-Fold 앙상블"

# 모델 선택 버튼 (가시성 높음)
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🎯 K-Fold 앙상블", key="kfold", use_container_width=True):
        st.session_state["selected_model"] = "K-Fold 앙상블"
with col2:
    if st.button("🚀 단일 YOLOv5 모델", key="single", use_container_width=True):
        st.session_state["selected_model"] = "단일 YOLOv5 모델"

# 세션 유지
model_type = st.session_state["selected_model"]
st.markdown(f"<h3 style='text-align: center; color: green;'>✅ 선택된 모델: {model_type}</h3>", unsafe_allow_html=True)

# 선택된 모델 로드
models = load_models() if model_type == "K-Fold 앙상블" else [load_single_model()]

# ✅ YOLOv5 클래스 이름 설정
CLASS_NAMES = ["extruded", "crack", "cutting", "side_stamped"]

# ✅ 이미지 전처리 함수
def preprocess_image(image, final_size=(500, 500), margin=20):
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
        resized_image = cv2.resize(cropped, final_size, interpolation=cv2.INTER_LINEAR)
        return resized_image
    else:
        return cv2.resize(image, final_size, interpolation=cv2.INTER_LINEAR)

# ✅ YOLOv5 입력 변환 함수
def prepare_image(image, img_size=640):
    img_resized, ratio, pad = letterbox(image, new_shape=img_size, auto=True)
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # HWC → CHW
    img_resized = np.ascontiguousarray(img_resized)
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가
    return image, img_tensor, ratio, pad

# ✅ YOLOv5 모델 예측 함수
def get_predictions(models, img, conf_thres=0.3, iou_thres=0.5):
    all_boxes, all_scores, all_labels = [], [], []

    for model in models:
        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

        if pred is not None and len(pred) > 0:
            all_boxes.append(pred[:, :4])
            all_scores.append(pred[:, 4])
            all_labels.append(pred[:, 5])

    if len(all_boxes) == 0:
        return [], [], []

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_boxes, all_scores, all_labels

# ✅ NMS 앙상블 함수
def ensemble_nms(boxes, scores, labels, iou_thres=0.5):
    keep_indices = ops.nms(boxes, scores, iou_thres)
    return boxes[keep_indices].cpu().numpy(), scores[keep_indices].cpu().numpy(), labels[keep_indices].cpu().numpy()

# 이미지 업로드
uploaded_files = st.file_uploader("📂 이미지를 업로드하세요", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# 세션 상태에서 현재 이미지 인덱스 저장 (초기값: 0)
if "image_index" not in st.session_state:
    st.session_state.image_index = 0

if uploaded_files:
    total_images = len(uploaded_files)

    if total_images > 1:
        selected_index = st.slider("이미지 선택", 0, total_images - 1, st.session_state.image_index)
        if selected_index != st.session_state.image_index:
            st.session_state.image_index = selected_index
            st.rerun()

    current_index = st.session_state.image_index
    uploaded_file = uploaded_files[current_index]
    image = Image.open(uploaded_file)
    image = np.array(image)

    # ✅ 전처리 적용
    processed_image = preprocess_image(image)

    # ✅ 모델 예측값 얻기
    _, img_tensor, ratio, pad = prepare_image(processed_image)
    boxes, scores, labels = get_predictions(models, img_tensor)

    # ✅ 결함 여부 체크
    has_defects = len(boxes) > 0

    if has_defects:
        # ✅ NMS 적용 (결함이 있는 경우에만)
        nms_boxes, nms_scores, nms_labels = ensemble_nms(boxes, scores, labels)

        # ✅ 바운딩 박스 위치 변환 적용
        nms_boxes = scale_boxes(img_tensor.shape[2:], torch.tensor(nms_boxes), processed_image.shape).round().numpy()

        # ✅ 바운딩 박스 색상 지정 (클래스별 다른 색상)
        COLORS = {"extruded": (0, 0, 255), "crack": (255, 0, 0), "cutting": (0, 255, 0), "side_stamped": (255, 255, 0)}

        # ✅ 결함 목록 저장
        defect_list = []

        # ✅ 결과 표시
        for box, score, label in zip(nms_boxes, nms_scores, nms_labels):
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Class {int(label)}"
            color = COLORS.get(class_name, (0, 255, 0))  # 기본 초록색

            # 바운딩 박스 그리기
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 3)

            # 텍스트 배경 박스 추가
            text = f"{class_name}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(processed_image, (x1, y1 - h - 5), (x1 + w + 10, y1), color, -1)

            # 텍스트 추가 (배경 위에 흰색 글씨)
            cv2.putText(processed_image, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # ✅ 결함 목록에 추가
            defect_list.append(f"- **{class_name}** (신뢰도: {score:.2f})")

        # ✅ 결함 메시지 표시
        st.markdown("<h3 style='text-align: center; color: red;'>⚠️ 결함이 검출되었습니다.</h3>", unsafe_allow_html=True)

        # ✅ 결함 목록 출력 (아래에 표시)
        for defect in defect_list:
            st.markdown(f"<h5 style='text-align: center; color: gray;'>{defect}</h5>", unsafe_allow_html=True)

    else:
        st.markdown("<h3 style='text-align: center; color: green;'>✅ 정상입니다.</h3>", unsafe_allow_html=True)

    st.image(processed_image, caption=f"Detection Results - {uploaded_file.name}", use_container_width=True)

    # **"이전" 및 "다음" 버튼 추가**
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