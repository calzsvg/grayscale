import cv2
import numpy as np
import os

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

def resize_image(file_path, target_width, target_height, output_path=None):
    if not isinstance(file_path, str):
        raise TypeError("파일 경로는 문자열이어야 합니다.")

    if not isinstance(target_width, int) or not isinstance(target_height, int) or target_width <= 0 or target_height <= 0:
        raise ValueError("너비와 높이는 1 이상의 정수여야 합니다.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {os.path.abspath(file_path)}")

    if output_path is None:
        file_root, file_ext = os.path.splitext(file_path)
        output_path = f"{file_root}_resized{file_ext}"

    try:
        file_array = np.fromfile(file_path, np.uint8)
        image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise Exception(f"이미지 디코딩 중 오류가 발생했습니다: {e}")

    if image is None:
        raise ValueError(f"이미지 로드에 실패했습니다: {file_path}")

    try:
        resized_image = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA
        )
    except Exception as e:
        raise Exception(f"이미지 크기 조정 중 오류가 발생했습니다: {e}")

    try:
        file_ext = os.path.splitext(output_path)[1]
        success, encoded_buffer = cv2.imencode(file_ext, resized_image)

        if success:
            with open(output_path, mode='w+b') as f:
                encoded_buffer.tofile(f)
        else:
            raise IOError("이미지 인코딩에 실패했습니다. 확장자와 데이터 형식을 확인해 주십시오.")

    except Exception as e:
        raise IOError(f"파일 저장에 실패했습니다: {output_path} - {e}")

def resize_folder(folder_path, target_width, target_height, output_folder=None):
    if not isinstance(folder_path, str):
        raise TypeError("폴더 경로는 문자열이어야 합니다.")

    if not isinstance(target_width, int) or not isinstance(target_height, int) or target_width <= 0 or target_height <= 0:
        raise ValueError("너비와 높이는 1 이상의 정수여야 합니다.")

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {os.path.abspath(folder_path)}")

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"유효한 폴더 경로가 아닙니다: {os.path.abspath(folder_path)}")

    if output_folder is None:
        output_folder = f"{folder_path}_resized"

    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    resized_count = 0

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(valid_extensions):
            continue

        input_path = os.path.join(folder_path, filename)
        file_root, file_ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{file_root}_resized{file_ext}")

        try:
            file_array = np.fromfile(input_path, np.uint8)
            image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

            if image is None:
                print(f"이미지 로드 실패 (건너뜀): {filename}")
                continue

            resized_image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA
            )

            success, encoded_buffer = cv2.imencode(file_ext, resized_image)

            if success:
                with open(output_path, mode='w+b') as f:
                    encoded_buffer.tofile(f)
                resized_count += 1
            else:
                raise IOError(f"인코딩에 실패했습니다: {filename}")

        except Exception as e:
            print(f"{filename} 처리 중 오류가 발생했습니다: {e}")

    print(f"리사이징 완료: {resized_count}개 이미지가 저장되었습니다 -> {output_folder}")

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