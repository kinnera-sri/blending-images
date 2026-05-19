import cv2
import numpy as np

pts_dst = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_dst.append((x, y))
        cv2.circle(im_display, (x, y), 5, (0, 255, 0), -1)
        # Force a refresh of the specific window
        cv2.imshow("Selection", im_display)

# Load images
im = cv2.imread("b99.png")         
obj = cv2.imread("apple.png")   

im_display = im.copy()
cv2.namedWindow("Selection")
cv2.setMouseCallback("Selection", select_points)

print("Click 4 points. Press 'q' to confirm and process.")

while True:
    cv2.imshow("Selection", im_display)
    key = cv2.waitKey(1) & 0xFF
    if len(pts_dst) == 4 or key == ord('q'):
        break

# Crucial: Destroy the selection window immediately to free memory 
# and prevent a "ghost" black window.
cv2.destroyWindow("Selection")

if len(pts_dst) == 4:
    # --- Transformation Logic ---
    h_obj, w_obj = obj.shape[:2]
    pts_src = np.array([[0, 0], [w_obj-1, 0], [w_obj-1, h_obj-1], [0, h_obj-1]], dtype=float)
    pts_dst_array = np.array(pts_dst, dtype=float)

    h_matrix, _ = cv2.findHomography(pts_src, pts_dst_array)
    obj_warped = cv2.warpPerspective(obj, h_matrix, (im.shape[1], im.shape[0]))

    mask = np.full(obj.shape, 255, dtype=np.uint8)
    mask_warped = cv2.warpPerspective(mask, h_matrix, (im.shape[1], im.shape[0]))

    # Calculate center using moments
    mask_gray = cv2.cvtColor(mask_warped, cv2.COLOR_BGR2GRAY)
    M = cv2.moments(mask_gray)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # Seamless Blend
    result = cv2.seamlessClone(obj_warped, im, mask_warped, center, cv2.MIXED_CLONE)

    # Show Final Result
    # cv2.imshow("Final Blend", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save 
    cv2.imwrite("seamless_blending.png", result)