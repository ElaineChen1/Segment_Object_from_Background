import cv2
import numpy as np
import mahotas

image = cv2.imread("puppy.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Simple Thresholding
(T, thresh) = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)
# Inverse Thresholding
(T, threshinv) = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV)
# Extract the foreground
fg_threshinv = cv2.bitwise_and(image, image, mask=threshinv)
cv2.imshow("Foreground", np.hstack([image, fg_threshinv]))
cv2.waitKey(0)

# Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(blur,
                                        255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        1491,
                                        2
                                        )
# Extract the foreground
fg_adap_thresh = cv2.bitwise_and(image, image, mask=adaptive_thresh)
cv2.imshow("Foreground A", np.hstack([image, fg_adap_thresh]))
cv2.waitKey(0)

# Gaussian Thresholding
gauss_thresh = cv2.adaptiveThreshold(blur,
                               255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               3301,
                               1
                              )
# Extract the foreground
fg_gauss_thresh = cv2.bitwise_and(image, image, mask=gauss_thresh)
cv2.imshow("Foreground G", np.hstack([image, fg_gauss_thresh]))
cv2.waitKey(0)

# Otsu Thresholding
T = mahotas.thresholding.otsu(blur)
thresh = blur.copy()
thresh[thresh >= T] = 255
thresh[thresh < 255] = 0
threshinv = cv2.bitwise_not(thresh)
# Extract the foreground
fg_otsu = cv2.bitwise_and(image, image, mask=threshinv)
cv2.imshow("Foreground Otsu", np.hstack([image, fg_otsu]))
cv2.waitKey(0)

# Riddler-Calvard Thresholding
T_RC = mahotas.thresholding.rc(blur)
thresh_RC = blur.copy()
thresh_RC[thresh_RC > T_RC] = 255
thresh_RC[thresh_RC < 255] = 0
threshinv_RC = cv2.bitwise_not(thresh_RC)
# Extract the foreground
fg_RC = cv2.bitwise_and(image, image, mask=threshinv_RC)
cv2.imshow("Foreground RC", np.hstack([image, fg_RC]))
cv2.waitKey(0)