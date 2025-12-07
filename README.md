# vidstream-grayscale (OSS Final Project)

Python 기반 이미지 및 영상 전처리(흑백 변환, 리사이즈, 흑백/컬러 판별, 배경제거)를 제공하는 간단한 유틸리티 모듈입니다.  
OpenCV, NumPy, MediaPipe를 사용해 이미지나 폴더 단위로 손쉽게 처리할 수 있습니다.  

---

## Features

이 프로젝트에서 제공하는 주요 기능은 다음과 같습니다.:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}

### 1. 이미지 흑백 변환

- `grayscale_image(file_path, output_path=None)`
- `grayscale_folder(folder_path, output_folder=None)`

이미지 1장 혹은 폴더 전체를 **grayscale(흑백)** 으로 변환합니다.  
입력 파일이 이미 흑백이면 그대로 저장하고, 컬러일 경우 자동으로 BGR → GRAY 변환을 수행합니다.

---

### 2. 이미지 리사이즈

- `resize_image(file_path, target_width, target_height, output_path=None)`
- `resize_folder(folder_path, target_width, target_height, output_folder=None)`

특징:

- 가로×세로 비율을 유지하면서 리사이즈
- 검은색 캔버스(배경) 위에 가운데 정렬해서 붙이는 방식(`image_resize`, `folder_resize`)
- 폴더 전체 일괄 처리 기능 제공

---

### 3. 흑백/컬러 이미지 판별 및 폴더 처리

- `is_gray(path, output_folder=None)`

단일 파일 또는 폴더를 입력했을 때, **흑백인지 컬러인지 판별**합니다.:contentReference[oaicite:8]{index=8}  

- 파일 1개: 콘솔에 “흑백 이미지입니다 / 컬러 이미지입니다” 출력  
- 폴더:
  - 전체 이미지 개수, 흑백/컬러 개수 출력
  - 흑백과 컬러가 섞여 있으면 콘솔에서 선택:
    1. 전체 흑백 변환
    2. 컬러 이미지만 다른 폴더로 분리
    3. 아무것도 하지 않음

---

### 4. 배경제거 (Person Segmentation)

- `remove_background(frame)`
- `remove_background_img(image_path)`:contentReference[oaicite:9]{index=9}  

MediaPipe Selfie Segmentation 모델을 활용해 사람을 분리하고,  
배경을 초록색으로 대체한 이미지를 생성합니다.

---

## Requirements

- Python >= 3.12:contentReference[oaicite:10]{index=10}
- NumPy
- OpenCV (opencv-python-headless)
- MediaPipe

`pyproject.toml`에 모든 의존성이 정의되어 있습니다.

```bash
# uv 사용 시
uv sync

# 또는 일반 가상환경 + pip
pip install -e .