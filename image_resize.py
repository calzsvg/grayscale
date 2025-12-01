import cv2
import numpy as np
import os

def image_resize(image_path, width, height, output_path=None):

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
        h, w = image.shape[:2]
        
        # 비율 계산
        scale = min(width / w, height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_temp = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 검은색 배경 생성
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 중앙 정렬
        x_offset = (width - new_w) // 2
        y_offset = (height - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_temp
        
        resized_image = canvas

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
                print(f"이미지 로드 실패 (건너뜀): {filename}")
                continue
            
            #비율 계산
            h, w = image.shape[:2]
        
            scale = min(target_width / w, target_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
        
            resized_temp = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 검은색 캔버스 생성
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # 중앙 좌표 계산
            x_offset = (target_width - new_w) // 2
            y_offset = (target_height - new_h) // 2
            
    
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_temp
            
            resized_image = canvas

            # 저장(한글 경로 지원)
            result, encoded_img = cv2.imencode(extension, resized_image)
            
            if result:
                with open(output_path, mode='w+b') as f:
                    encoded_img.tofile(f)
                resized_count += 1
            else:
                raise IOError(f"인코딩 실패: {filename} - 확장자를 확인해 주세요")

        except Exception as e:
            print(f"{filename} 처리 중 오류 발생: {e}")

    print(f"리사이징 완료 : {resized_count}개 이미지 저장됨 → {output_folder}")
