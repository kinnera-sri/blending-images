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

feather_radius = 5

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
# OPEN DEBUG LOG FILE
# ============================================================

log_path = "blending-images/result_images/debug_log.txt"

log_file = open(log_path, "w")
# ============================================================
# DEBUG EDGE PIXEL ANALYSIS
# ============================================================

debug_x = 126
debug_y = 247

log_file.write("\n================ EDGE DEBUG ================\n\n")

# ------------------------------------------------------------
# RAW PIXELS
# ------------------------------------------------------------

src_pixel = warped_src[debug_y, debug_x]

dst_pixel = dst[debug_y, debug_x]

blend_pixel = blended[debug_y, debug_x]

alpha_pixel = alpha[debug_y, debug_x]

log_file.write(
    f"Pixel Location: ({debug_x}, {debug_y})\n\n"
)

log_file.write("SOURCE PIXEL (logo):\n")
log_file.write(f"{src_pixel}\n\n")

log_file.write("DESTINATION PIXEL:\n")
log_file.write(f"{dst_pixel}\n\n")

log_file.write("ALPHA VALUE:\n")
log_file.write(f"{alpha_pixel}\n\n")

log_file.write("FINAL BLENDED PIXEL:\n")
log_file.write(f"{blend_pixel}\n\n")


# ------------------------------------------------------------
# MANUAL BLEND VERIFICATION
# ------------------------------------------------------------

manual = (
    alpha_pixel * src_pixel.astype(np.float32)
    +
    (1.0 - alpha_pixel)
    * dst_pixel.astype(np.float32)
)

log_file.write("MANUAL COMPUTED BLEND:\n")
log_file.write(f"{manual.astype(np.uint8)}\n\n")


# ------------------------------------------------------------
# ALPHA PROFILE
# ------------------------------------------------------------

log_file.write(
    "============= ALPHA PROFILE =============\n\n"
)

for offset in range(-10, 11):

    xx = debug_x + offset

    a = alpha[debug_y, xx][0]

    log_file.write(
        f"x={xx:4d} | alpha={a:.6f}\n"
    )


# ------------------------------------------------------------
# SOURCE / DEST DIFFERENCE
# ------------------------------------------------------------

log_file.write(
    "\n============= COLOR DIFFERENCE =============\n\n"
)

difference = (
    src_pixel.astype(np.int32)
    -
    dst_pixel.astype(np.int32)
)

log_file.write(
    f"Source - Destination:\n{difference}\n\n"
)


# ------------------------------------------------------------
# VISUALIZE LOCAL REGION
# ------------------------------------------------------------

region_size = 60

x1 = max(debug_x - region_size, 0)
x2 = min(debug_x + region_size, dst.shape[1])

y1 = max(debug_y - region_size, 0)
y2 = min(debug_y + region_size, dst.shape[0])

src_crop = warped_src[y1:y2, x1:x2].copy()

dst_crop = dst[y1:y2, x1:x2].copy()

mask_crop = soft_mask[y1:y2, x1:x2].copy()

blend_crop = blended[y1:y2, x1:x2].copy()

cx = debug_x - x1
cy = debug_y - y1

for img in [src_crop, dst_crop, blend_crop]:

    cv2.circle(
        img,
        (cx, cy),
        4,
        (0,0,255),
        -1
    )

cv2.imwrite(
    "blending-images/result_images/debug_src_crop.png",
    src_crop
)

cv2.imwrite(
    "blending-images/result_images/debug_dst_crop.png",
    dst_crop
)

cv2.imwrite(
    "blending-images/result_images/debug_mask_crop.png",
    mask_crop
)

cv2.imwrite(
    "blending-images/result_images/debug_blend_crop.png",
    blend_crop
)

log_file.write(
    "Saved debug crop visualizations.\n"
)

log_file.write(
    "\n===========================================\n"
)


# ============================================================
# SAVE RESULTS
# ============================================================

cv2.imwrite("blending-images/result_images/warped_logo_feather_office_pepsi.png", warped_src)
cv2.imwrite("blending-images/result_images/warped_mask_feather_office_pepsi.png", warped_mask)
cv2.imwrite("blending-images/result_images/soft_mask_feather_office_pepsi.png", soft_mask)
cv2.imwrite("blending-images/result_images/feather_blended_office_pepsi.png", blended)

print("Done.")
log_file.close()

# cv2.imshow("Final Blend", blended)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# (x, y): edge pixel, need to visualize every step and values to see why the ghost effect happens at the edge of the logo
