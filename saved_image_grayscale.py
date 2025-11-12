import cv2


image = cv2.imread('sample.jpg')

# 2. 흑백으로 변환
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


cv2.imshow('Original Image', image)
cv2.imshow('Gray Image', gray_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. 저장하기 
cv2.imwrite('gray_image.jpg', gray_image)
