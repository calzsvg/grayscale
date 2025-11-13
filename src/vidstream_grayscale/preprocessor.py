import cv2
import numpy as np
import os


def grayScaling(frame):
    
    h, w, c = frame.shape
    if c == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif c == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        gray = frame
    
    return gray

def image_grayscale(image_path, output_path=None):


    #입력값 검증
    if not isinstance(image_path, str):
        raise TypeError("파일 경로는 문자열 이여야 합니다")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {os.path.abspath(image_path)}")

    #사용자가 출력 사진 이름을 지정하지 않을시
    if output_path is None:
        #확장자(.jpg)와 앞부분(경로+파일이름)을 분리합니다.
        root, extension = os.path.splitext(image_path)
        #f-sting을 사용해 '_gray'와 붙여 합칩니다.
        output_path = f"{root}_gray{extension}"

    try:
        img_array = np.fromfile(image_path, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise Exception(f"이미지 디코딩 오류: {e}")

    if image is None:
        raise ValueError(f"이미지 로드 실패 :{image_path}")
    

    # 2. 흑백으로 변환
    #이미 흑백 사진을 받을시 그냥 gray = image
    if len(image.shape) ==2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #결과 저장 (한글 경로 지원)
    try:
        extension = os.path.splitext(output_path)[1]
        result , encoded_img = cv2.imencode(extension , gray)

        if result:
            with open(output_path, mode ='w+b') as f:
                encoded_img.tofile(f)
        else:
            raise IOError("인코딩 실패, 확장자를 확인하거나, 이미지 데이터를 확인해 주세요")
        
    except Exception as e:
        raise IOError(f" 저장 실패 {output_path}- {e}")
    

#image_grayscale("sample.jpg") 잘 돌아가는것 확인.