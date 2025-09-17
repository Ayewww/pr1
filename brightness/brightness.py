import cv2 as cv
import numpy as np

def brightness1():
    src = cv.imread("lenna.bmp", cv.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    dst = cv.add(src, 100)  # OpenCV의 add 함수로 밝기 조정

    cv.imshow("src", src)
    cv.imshow("dst", dst)
    cv.waitKey()
    cv.destroyAllWindows()

def brightness2():
    src = cv.imread("lenna.bmp", cv.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    dst = np.zeros_like(src, dtype=np.uint8)

    for j in range(src.shape[0]):
        for i in range(src.shape[1]):
            dst[j, i] = src[j, i] + 100

    cv.imshow("src", src)
    cv.imshow("dst", dst)
    cv.waitKey()
    cv.destroyAllWindows()

def brightness3():
    src = cv.imread("lenna.bmp", cv.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    dst = np.zeros_like(src, dtype=np.uint8)

    for j in range(src.shape[0]):
        for i in range(src.shape[1]):
            dst[j, i] = np.clip(src[j, i] + 100, 0, 255)  # saturate_cast 대체

    cv.imshow("src", src)
    cv.imshow("dst", dst)
    cv.waitKey()
    cv.destroyAllWindows()

def brightness4():
    def on_brightness(pos):
        dst = cv.add(src, pos)
        cv.imshow("dst", dst)

    src = cv.imread("lenna.bmp", cv.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    cv.namedWindow("dst")
    cv.createTrackbar("Brightness", "dst", 0, 100, on_brightness)
    on_brightness(0)

    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == "__main__":
    brightness1()
    brightness2()
    brightness3()
    brightness4()