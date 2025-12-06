import cv2
import numpy as np
import os
import mediapipe as mp

def grayscale_frame(frame):
    h, w, c = frame.shape
    if c == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif c == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        return frame

def grayscale_image(file_path, output_path=None):
    if not isinstance(file_path, str):
        raise TypeError("파일 경로는 문자열이어야 합니다.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {os.path.abspath(file_path)}")

    if output_path is None:
        file_root, file_ext = os.path.splitext(file_path)
        output_path = f"{file_root}_gray{file_ext}"

    try:
        file_array = np.fromfile(file_path, np.uint8)
        image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise Exception(f"이미지 디코딩 중 오류가 발생했습니다: {e}")

    if image is None:
        raise ValueError(f"이미지 로드에 실패했습니다: {file_path}")

    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        file_ext = os.path.splitext(output_path)[1]
        success, encoded_buffer = cv2.imencode(file_ext, gray_image)

        if success:
            with open(output_path, mode='w+b') as f:
                encoded_buffer.tofile(f)
        else:
            raise IOError("이미지 인코딩에 실패했습니다. 확장자와 데이터 형식을 확인해 주십시오.")

    except Exception as e:
        raise IOError(f"파일 저장에 실패했습니다: {output_path} - {e}")

def grayscale_folder(folder_path, output_folder=None):
    if not isinstance(folder_path, str):
        raise TypeError("폴더 경로는 문자열이어야 합니다.")

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {os.path.abspath(folder_path)}")

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"유효한 폴더 경로가 아닙니다: {os.path.abspath(folder_path)}")

    if output_folder is None:
        output_folder = f"{folder_path}_gray"

    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    converted_count = 0

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(valid_extensions):
            continue

        input_path = os.path.join(folder_path, filename)
        file_root, file_ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{file_root}_gray{file_ext}")

        try:
            file_array = np.fromfile(input_path, np.uint8)
            image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

            if image is None:
                print(f"이미지 로드 실패 (건너뜀): {filename}")
                continue

            if len(image.shape) == 2:
                gray_image = image
            else:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            success, encoded_buffer = cv2.imencode(file_ext, gray_image)
            if success:
                with open(output_path, mode='w+b') as f:
                    encoded_buffer.tofile(f)
                converted_count += 1
            else:
                raise IOError(f"인코딩에 실패했습니다: {filename}")

        except Exception as e:
            print(f"[ERROR] {filename} 처리 중 오류가 발생했습니다: {e}")

    print(f"*** 변환 완료: {converted_count}개 이미지가 저장되었습니다 -> {output_folder} ***")


def resize_image(image_path, width, height, output_path=None):

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



def resize_folder(folder_path, target_width, target_height, output_folder=None):
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

def is_gray(path, output_folder=None):

    import shutil 
    
    #경로 확인
    if not isinstance(path, str):
        raise TypeError("경로는 문자열이어야 합니다.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {os.path.abspath(path)}")

    #흑백 판별 내부 함수
    def _check_image_gray(image_array):
        if len(image_array.shape) < 3 or image_array.shape[2] == 1:
            return True
        b, g, r = cv2.split(image_array)
        if np.array_equal(b, g) and np.array_equal(g, r):
            return True
        return False

    #파일 처리
    if os.path.isfile(path):
        try:
            file_array = np.fromfile(path, np.uint8)
            image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
            if image is None: raise ValueError(f"이미지 로드 실패: {path}")

            if _check_image_gray(image):
                print(f"결과: 흑백 이미지입니다.")
                return True
            else:
                print(f"결과: 컬러 이미지입니다.")
                return False
        except Exception as e:
            raise Exception(f"파일 검사 오류: {e}")

    #폴더 처리
    elif os.path.isdir(path):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        gray_count = 0
        color_files = [] 

        for filename in os.listdir(path):
            if not filename.lower().endswith(valid_extensions):
                continue

            input_path = os.path.join(path, filename)

            try:
                file_array = np.fromfile(input_path, np.uint8)
                image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

                if image is None:
                    continue

                if _check_image_gray(image):
                    gray_count += 1
                else:
                    color_files.append(input_path)

            except Exception as e:
                print(f"오류 {filename} 읽기 실패: {e}")

        color_count = len(color_files)
        total_count = gray_count + color_count

        print(f"\n-------------")
        print(f"총 이미지: {total_count}장")
        print(f"흑백: {gray_count}장 / 컬러: {color_count}장")

        #혼합된 경우 옵션
        if gray_count > 0 and color_count > 0:
            print(f"\n***흑백과 컬러가 섞여 있습니다. 처리 방법을 선택하세요.***")
            print("1. 전체 흑백 변환")
            print("2. 컬러 분리")
            print("3. 원본 유지")
            
            while True:
                choice = input(">>> 선택 (1/2/3): ").strip()

                if choice == '1':
                    grayscale_folder(path, output_folder) 
                    break

                elif choice == '2':
                    if output_folder is None:
                        output_folder = f"{path}_color"
                    
                    os.makedirs(output_folder, exist_ok=True)
                    
                    moved_count = 0
                    for src_path in color_files:
                        try:
                            filename = os.path.basename(src_path)
                            dst_path = os.path.join(output_folder, filename)
                            
                            # 파일 이동
                            shutil.move(src_path, dst_path)
                            moved_count += 1
                        except Exception as e:
                            print(f"오류 이동 실패 ({filename}): {e}")
                    
                    print(f"분리 완료: {moved_count}장의 컬러 이미지가 '{output_folder}'(으)로 분리되었습니다.")
                    print(f"원본 폴더에는 흑백 이미지만 남았습니다.")
                    break

                elif choice == '3':
                    print("\n원본을 그대로 유지합니다.")
                    break

                else:
                    print("잘못된 입력입니다. 1, 2, 3 중 하나를 입력해주세요.")

        elif color_count == total_count:
             print("결과: 모든 이미지가 컬러입니다.")
        elif gray_count == total_count:
             print("결과: 모든 이미지가 흑백입니다.")

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def remove_background(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = segmenter.process(frame_rgb)
    mask = results.segmentation_mask
    condition = np.stack((mask,) * 3, axis=-1) > 0.5
    bg_image = np.zeros(frame.shape, dtype=np.uint8)
    bg_image[:] = (0, 255, 0)
    output_image = np.where(condition, frame, bg_image)
    return output_image

def remove_background_img(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {image_path}")

    try:
        file_array = np.fromfile(image_path, np.uint8)
        image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise Exception(f"이미지 디코딩 중 오류가 발생했습니다: {e}")
    
    if image is None:
        raise ValueError(f"파일 형식을 확인하세요: {image_path}")

    result_image = remove_background(image)
    
    return result_image