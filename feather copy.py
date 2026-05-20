import cv2
import numpy as np


# ============================================================
# LOAD IMAGES
# ============================================================

src = cv2.imread(
    "blending-images/main_images/pepsi_logo.png",
    cv2.IMREAD_UNCHANGED
)

# src = cv2.imread(
#     "blending-images/main_images/lego.png",
#     cv2.IMREAD_UNCHANGED
# )

dst = cv2.imread("blending-images/main_images/office.png")
# dst = cv2.imread("blending-images/main_images/b99_2.png")
# dst = cv2.imread("blending-images/main_images/b99.png")


if src is None or dst is None:
    raise ValueError("Could not load images")

# split logo + alpha
if src.shape[2] == 4:

    logo = src[:, :, :3]

    mask = src[:, :, 3]

else:
    raise ValueError(
        "Logo PNG must contain alpha channel"
    )

# ============================================================
# SELECT 4 DESTINATION POINTS
# ============================================================

points = []
display = dst.copy()


def mouse_callback(event, x, y, flags, param):

    global points, display

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append([x, y])

        # draw clicked point
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

        # draw order number
        cv2.putText(
            display,
            str(len(points)),
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Select 4 Points", display)


cv2.namedWindow("Select 4 Points")
cv2.setMouseCallback("Select 4 Points", mouse_callback)

print("Click 4 points in clockwise order:")
print("Top-left -> Top-right -> Bottom-right -> Bottom-left")

while True:

    cv2.imshow("Select 4 Points", display)

    key = cv2.waitKey(1) & 0xFF

    # press r to reset
    if key == ord('r'):
        points = []
        display = dst.copy()

    # press q to quit
    elif key == ord('q'):
        break

    # automatically stop after 4 points
    if len(points) == 4:
        break

cv2.destroyAllWindows()

if len(points) != 4:
    raise ValueError("You must select exactly 4 points")


# ============================================================
# HOMOGRAPHY / PERSPECTIVE WARP
# ============================================================

h_src, w_src = src.shape[:2]

src_pts = np.float32([
    [0, 0],
    [w_src - 1, 0],
    [w_src - 1, h_src - 1],
    [0, h_src - 1]
])

dst_pts = np.float32(points)

# perspective transform matrix
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# warp logo and mask
warped_src = cv2.warpPerspective(
    logo,
    H,
    (dst.shape[1], dst.shape[0])
)
# print("Warped source shape:", warped_src.shape)

warped_mask = cv2.warpPerspective(
    mask,
    H,
    (dst.shape[1], dst.shape[0])
)
# print("Warped mask shape:", warped_mask.shape)


# ============================================================
# CREATE FEATHERED MASK
# ============================================================

feather_radius = 27

soft_mask = cv2.GaussianBlur(
    warped_mask,
    (feather_radius, feather_radius),
    0
)

alpha = soft_mask.astype(np.float32) / 255.0

alpha = cv2.merge([alpha, alpha, alpha])


# ============================================================
# FEATHER BLENDING
# ============================================================

src_f = warped_src.astype(np.float32)
dst_f = dst.astype(np.float32)

blended = (
    alpha * src_f +
    (1.0 - alpha) * dst_f
)

blended = np.clip(blended, 0, 255).astype(np.uint8)


# ============================================================
# SAVE RESULTS
# ============================================================

cv2.imwrite("blending-images/result_images/warped_logo_feather_office_pepsi.png", warped_src)
cv2.imwrite("blending-images/result_images/warped_mask_feather_office_pepsi.png", warped_mask)
cv2.imwrite("blending-images/result_images/soft_mask_feather_office_pepsi.png", soft_mask)
cv2.imwrite("blending-images/result_images/feather_blended_office_pepsi.png", blended)

print("Done.")

# cv2.imshow("Final Blend", blended)
# cv2.waitKey(0)
# cv2.destroyAllWindows()