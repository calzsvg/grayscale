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