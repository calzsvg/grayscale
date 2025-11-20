import cv2
import numpy as np
import os


def vidstr_grayscaling(frame):
    
    h, w, c = frame.shape
    if c == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif c == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        gray = frame
    
    return gray

def img_grayscale(image_path, output_path=None):


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


#이미지 폴더 흑백 변환 함수

def fol_grayscale(folder_path, output_folder=None):
    
    #주어진 폴더 내의 모든 이미지를 흑백으로 변환.
    #image_grayscale()과 동일한 예외 처리 및 기능 포함, image_grayscale()과 최대한 비슷한 구조로 구현.


    # 입력값 검증
    if not isinstance(folder_path, str):
        raise TypeError("폴더 경로는 문자열이어야 합니다.")

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {os.path.abspath(folder_path)}")

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"폴더 경로가 아닙니다: {os.path.abspath(folder_path)}")

    # 출력 폴더 경로 설정
    if output_folder is None:
        output_folder = f"{folder_path}_gray"

    os.makedirs(output_folder, exist_ok=True)

    # 지원하는 이미지 확장자
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    converted_count = 0

    # 폴더 내 파일 순회
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(valid_ext):
            continue  # 이미지 파일만 처리

        input_path = os.path.join(folder_path, filename)
        root, extension = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{root}_gray{extension}")

        try:
            # 한글 경로를 고려한 로드
            img_array = np.fromfile(input_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image is None:
                print(f"[WARN] 이미지 로드 실패: {filename}")
                continue

            # 흑백 변환
            if len(image.shape) == 2:
                gray = image
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 저장(한글 경로 지원)
            result, encoded_img = cv2.imencode(extension, gray)
            if result:
                with open(output_path, mode='w+b') as f:
                    encoded_img.tofile(f)
                converted_count += 1
            else:
                raise IOError(f"인코딩 실패: {filename}")

        except Exception as e:
            print(f"[ERROR] {filename} 처리 중 오류 발생: {e}")

    print(f"***변환 완료 : {converted_count}개 이미지 저장됨 → {output_folder}***")


def img_resize(image_path, width, height, output_path=None):

    # 입력값 검증 
    if not isinstance(image_path, str):
        raise TypeError("파일 경로는 문자열 이여야 합니다")
        
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("너비와 높이는 1 이상의 정수여야 합니다.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {os.path.abspath(image_path)}")

    #사용자가 출력 사진 이름을 지정하지 않을시
    if output_path is None:
        #확장자(.jpg)와 앞부분(경로+파일이름)을 분리합니다.
        root, extension = os.path.splitext(image_path)
        #f-sting을 사용해 '_resized_'와 붙여 합칩니다.
        output_path = f"{root}_resized{extension}"
        

    try:
        img_array = np.fromfile(image_path, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise Exception(f"이미지 디코딩 오류: {e}")

    if image is None:
        raise ValueError(f"이미지 로드 실패 :{image_path}")
    
    # 이미지 크기 조정
    try:
        resized_image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA
        )
    except Exception as e:
        raise Exception(f"이미지 리사이징 오류: {e}")


    # 결과 저장 (한글 경로 지원)
    try:
        result , encoded_img = cv2.imencode(extension , resized_image)

        if result:
            with open(output_path, mode ='w+b') as f:
                encoded_img.tofile(f)
        else:
            raise IOError("인코딩 실패, 확장자를 확인하거나, 이미지 데이터를 확인해 주세요")
        
    except Exception as e:
        raise IOError(f" 저장 실패 {output_path}- {e}")
    

#image_resize("sample.jpg",100,100) #잘 돌아가는것 확인



def folder_resize(folder_path, target_width, target_height, output_folder=None):
    #주어진 폴더 내의 모든 이미지를 리사이징

    # 입력값 검증
    if not isinstance(folder_path, str):
        raise TypeError("폴더 경로는 문자열이어야 합니다.")
        
    if not isinstance(target_width, int) or not isinstance(target_height, int) or target_width <= 0 or target_height <= 0:
        raise ValueError("너비와 높이는 1 이상의 정수여야 합니다.")

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {os.path.abspath(folder_path)}")

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"폴더 경로가 아닙니다: {os.path.abspath(folder_path)}")

    # 출력 폴더 경로 설정
    if output_folder is None:
        output_folder = f"{folder_path}_resized"

    os.makedirs(output_folder, exist_ok=True)
    

    # 지원하는 이미지 확장자
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    resized_count = 0
    

    # 폴더 내 파일 순회
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(valid_ext):
            continue  # 이미지 파일만 처리

        input_path = os.path.join(folder_path, filename)
        root, extension = os.path.splitext(filename)
        output_filename = f"{root}_resized{extension}"
        output_path = os.path.join(output_folder, output_filename)

        try:
            # 한글 경로를 고려한 로드
            img_array = np.fromfile(input_path, np.uint8)

            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image is None:
                print(f"[WARN] 이미지 로드 실패 (건너뜀): {filename}")
                continue

            # 이미지 크기 조정
            resized_image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA
            )

            # 저장(한글 경로 지원)
            result, encoded_img = cv2.imencode(extension, resized_image)
            
            if result:
                with open(output_path, mode='w+b') as f:
                    encoded_img.tofile(f)
                resized_count += 1
            else:
                raise IOError(f"인코딩 실패: {filename} - 확장자를 확인해 주세요")

        except Exception as e:
            print(f"[ERROR] {filename} 처리 중 오류 발생: {e}")

    print(f"***리사이징 완료 : {resized_count}개 이미지 저장됨 → {output_folder}***")