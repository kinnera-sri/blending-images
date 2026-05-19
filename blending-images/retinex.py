import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

SOURCE_PATH = "lego.png"
DEST_PATH = "b99.png"

OUTPUT_NAME = "retinex_blended.png"

BLUR_SIGMA = 41

FEATHER_RADIUS = 31


# ============================================================
# LOAD IMAGES
# ============================================================

# Load PNG with alpha
src = cv2.imread(
    SOURCE_PATH,
    cv2.IMREAD_UNCHANGED
)

dst = cv2.imread(DEST_PATH)

if src is None or dst is None:
    raise ValueError("Could not load images")


# ============================================================
# SPLIT LOGO + MASK
# ============================================================

if src.shape[2] != 4:
    raise ValueError(
        "Source image must contain alpha channel"
    )

logo = src[:, :, :3]
mask = src[:, :, 3]


# ============================================================
# SELECT 4 POINTS
# ============================================================

points = []
display = dst.copy()


def mouse_callback(event, x, y, flags, param):

    global points, display

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append([x, y])

        cv2.circle(
            display,
            (x, y),
            5,
            (0,0,255),
            -1
        )

        cv2.putText(
            display,
            str(len(points)),
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.imshow("Select 4 Points", display)


cv2.namedWindow("Select 4 Points")
cv2.setMouseCallback(
    "Select 4 Points",
    mouse_callback
)

print("\nClick 4 points CLOCKWISE:")
print("Top-left -> Top-right -> Bottom-right -> Bottom-left\n")

while True:

    cv2.imshow("Select 4 Points", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):

        points = []
        display = dst.copy()

    elif key == ord('q'):

        break

    if len(points) == 4:

        break

cv2.destroyAllWindows()

if len(points) != 4:
    raise ValueError("Need exactly 4 points")


# ============================================================
# PERSPECTIVE WARP
# ============================================================

h_src, w_src = logo.shape[:2]

src_pts = np.float32([
    [0,0],
    [w_src-1,0],
    [w_src-1,h_src-1],
    [0,h_src-1]
])

dst_pts = np.float32(points)

H = cv2.getPerspectiveTransform(
    src_pts,
    dst_pts
)

warped_logo = cv2.warpPerspective(
    logo,
    H,
    (dst.shape[1], dst.shape[0])
)

warped_mask = cv2.warpPerspective(
    mask,
    H,
    (dst.shape[1], dst.shape[0])
)


# ============================================================
# FEATHER MASK
# ============================================================

soft_mask = cv2.GaussianBlur(
    warped_mask,
    (FEATHER_RADIUS, FEATHER_RADIUS),
    0
)

alpha = soft_mask.astype(np.float32) / 255.0

alpha_3 = cv2.merge([alpha, alpha, alpha])


# ============================================================
# CONVERT TO FLOAT
# ============================================================

logo_f = warped_logo.astype(np.float32) / 255.0
dst_f = dst.astype(np.float32) / 255.0


# ============================================================
# RETINEX DECOMPOSITION
# ============================================================

# estimate illumination
illum_logo = cv2.GaussianBlur(
    logo_f,
    (0,0),
    BLUR_SIGMA
)

illum_dst = cv2.GaussianBlur(
    dst_f,
    (0,0),
    BLUR_SIGMA
)

# avoid division by zero
eps = 1e-3

# reflectance
reflect_logo = logo_f / (illum_logo + eps)

reflect_dst = dst_f / (illum_dst + eps)


# ============================================================
# RETINEX BLENDING
# ============================================================

# Blend reflectance
reflect_blend = (
    alpha_3 * reflect_logo +
    (1.0 - alpha_3) * reflect_dst
)

# Blend illumination
illum_blend = (
    alpha_3 * illum_dst +
    (1.0 - alpha_3) * illum_dst
)

# reconstruct
result = reflect_blend * illum_blend

result = np.clip(result, 0, 1)

result_u8 = (result * 255).astype(np.uint8)


# ============================================================
# VISUALIZATION
# ============================================================

illum_logo_vis = np.clip(
    illum_logo * 255,
    0,
    255
).astype(np.uint8)

reflect_logo_vis = np.clip(
    reflect_logo * 40,
    0,
    255
).astype(np.uint8)

alpha_vis = (alpha * 255).astype(np.uint8)


# ============================================================
# SAVE OUTPUTS
# ============================================================

cv2.imwrite(
    "01_warped_logo.png",
    warped_logo
)

cv2.imwrite(
    "02_warped_mask.png",
    warped_mask
)

cv2.imwrite(
    "03_soft_mask.png",
    alpha_vis
)

cv2.imwrite(
    "04_logo_illumination.png",
    illum_logo_vis
)

cv2.imwrite(
    "05_logo_reflectance.png",
    reflect_logo_vis
)

cv2.imwrite(
    OUTPUT_NAME,
    result_u8
)

print("\nSaved outputs:")
print(" - warped logo")
print(" - warped mask")
print(" - soft mask")
print(" - illumination estimate")
print(" - reflectance estimate")
print(" - final Retinex blend")


# ============================================================
# SHOW FINAL
# ============================================================

# cv2.imshow("Retinex Blend", result_u8)
# cv2.waitKey(0)
# cv2.destroyAllWindows()