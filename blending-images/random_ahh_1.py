import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

SOURCE_PATH = "lego.png"
DEST_PATH = "office.png"

FEATHER_RADIUS = 31

TEXTURE_STRENGTH = 0.15


# ============================================================
# LOAD IMAGES
# ============================================================

src = cv2.imread(
    SOURCE_PATH,
    cv2.IMREAD_UNCHANGED
)

dst = cv2.imread(DEST_PATH)

if src is None or dst is None:
    raise ValueError("Could not load images")


# ============================================================
# SPLIT LOGO + ALPHA
# ============================================================

if src.shape[2] != 4:
    raise ValueError("PNG must contain alpha channel")

logo = src[:, :, :3]
mask = src[:, :, 3]

# ============================================================
# SELECT 4 POINTS
# ============================================================

points = []

display = dst.copy()

WINDOW_NAME = "Select 4 Points"


def mouse_callback(event, x, y, flags, param):

    global points, display

    if event == cv2.EVENT_LBUTTONDOWN:

        # prevent extra clicks
        if len(points) >= 4:
            return

        points.append([x, y])

        # redraw fresh image
        display = dst.copy()

        # draw all selected points
        for idx, p in enumerate(points):

            px, py = p

            cv2.circle(
                display,
                (px, py),
                8,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display,
                str(idx + 1),
                (px + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


cv2.namedWindow(WINDOW_NAME)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)

print("\nClick 4 points:")
print("Top-left -> Top-right -> Bottom-right -> Bottom-left")
print("Press 'r' to reset")
print("Press 'q' to quit\n")


while True:

    cv2.imshow(WINDOW_NAME, display)

    key = cv2.waitKey(20) & 0xFF

    # reset
    if key == ord('r'):

        points = []

        display = dst.copy()

    # quit
    elif key == ord('q'):

        cv2.destroyAllWindows()

        raise SystemExit

    # automatically finish
    if len(points) == 4:

        break


cv2.destroyAllWindows()

cv2.waitKey(1)

print("\nSelected points:")
print(points)    

# ============================================================
# HOMOGRAPHY + WARP
# ============================================================

h_logo, w_logo = logo.shape[:2]

src_pts = np.float32([
    [0, 0],
    [w_logo - 1, 0],
    [w_logo - 1, h_logo - 1],
    [0, h_logo - 1]
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

cv2.imwrite(
    "debug_warped_logo.png",
    warped_logo
)

cv2.imwrite(
    "debug_warped_mask.png",
    warped_mask
)

print("Saved warped debug outputs")


# ============================================================
# FEATHER MASK
# ============================================================

soft_mask = cv2.GaussianBlur(
    warped_mask,
    (FEATHER_RADIUS, FEATHER_RADIUS),
    0
)

alpha = soft_mask.astype(np.float32) / 255.0

alpha_3 = cv2.merge([
    alpha,
    alpha,
    alpha
])


# ============================================================
# SHARPNESS MATCHING
# ============================================================

dst_gray = cv2.cvtColor(
    dst,
    cv2.COLOR_BGR2GRAY
)

logo_gray = cv2.cvtColor(
    warped_logo,
    cv2.COLOR_BGR2GRAY
)

sharp_dst = cv2.Laplacian(
    dst_gray,
    cv2.CV_32F
).var()

sharp_logo = cv2.Laplacian(
    logo_gray,
    cv2.CV_32F
).var()

print(f"Destination sharpness: {sharp_dst:.2f}")
print(f"Logo sharpness: {sharp_logo:.2f}")


# blur logo if too sharp
if sharp_logo > sharp_dst:

    ratio = sharp_logo / (sharp_dst + 1e-5)

    sigma = min(
        np.log(ratio + 1.0),
        5
    )

    warped_logo = cv2.GaussianBlur(
        warped_logo,
        (0,0),
        sigma
    )

    print(f"Applied blur sigma: {sigma:.2f}")


# ============================================================
# BRIGHTNESS / CONTRAST MATCHING
# ============================================================

mask_binary = warped_mask > 10

logo_pixels = warped_logo[
    mask_binary
]

dst_pixels = dst[
    mask_binary
]

logo_mean = logo_pixels.mean(axis=0)
logo_std = logo_pixels.std(axis=0)

dst_mean = dst_pixels.mean(axis=0)
dst_std = dst_pixels.std(axis=0)

warped_logo_f = warped_logo.astype(np.float32)

for c in range(3):

    warped_logo_f[:,:,c] = (

        (warped_logo_f[:,:,c] - logo_mean[c])

        * (dst_std[c] / (logo_std[c] + 1e-5))

        + dst_mean[c]
    )

warped_logo_f = np.clip(
    warped_logo_f,
    0,
    255
)


# # ============================================================
# # FEATHER BLENDING
# # ============================================================

# src_f = warped_logo_f / 255.0

# dst_f = dst.astype(np.float32) / 255.0

# blended = (

#     alpha_3 * src_f +

#     (1.0 - alpha_3) * dst_f
# )

# # ============================================================
# # MULTIPLY BLENDING
# # ============================================================

# logo_gray = cv2.cvtColor(
#     warped_logo.astype(np.uint8),
#     cv2.COLOR_BGR2GRAY
# )

# logo_norm = logo_gray.astype(np.float32) / 255.0

# # invert if needed
# logo_norm = 1.0 - logo_norm

# # expand channels
# logo_norm = cv2.merge([
#     logo_norm,
#     logo_norm,
#     logo_norm
# ])

# dst_f = dst.astype(np.float32) / 255.0

# # logo strength
# strength = 0.35

# # multiplicative blending
# blended = dst_f * (
#     1.0 - strength * alpha_3 * logo_norm
# )

# blended = np.clip(
#     blended,
#     0,
#     1
# )

# ============================================================
# HYBRID BLENDING
# ============================================================

dst_f = dst.astype(np.float32) / 255.0

logo_f = warped_logo.astype(np.float32) / 255.0

# ------------------------------------------------------------
# 1. SOFT ALPHA COMPOSITING
# ------------------------------------------------------------

base_blend = (

    alpha_3 * logo_f * 0.55 +

    (1.0 - alpha_3 * 0.55) * dst_f
)

# ------------------------------------------------------------
# 2. DESTINATION TEXTURE EXTRACTION
# ------------------------------------------------------------

blurred_dst = cv2.GaussianBlur(
    dst_f,
    (0,0),
    3
)

texture = dst_f - blurred_dst

# ------------------------------------------------------------
# 3. TEXTURE MODULATION
# ------------------------------------------------------------

base_blend += (
    texture
    * alpha_3
    * 0.18
)

# ------------------------------------------------------------
# 4. LOCAL CONTRAST REDUCTION
# ------------------------------------------------------------

# logos on rough surfaces lose contrast naturally

mean = cv2.GaussianBlur(
    base_blend,
    (0,0),
    5
)

base_blend = (
    0.75 * base_blend +
    0.25 * mean
)

# ------------------------------------------------------------
# 5. VERY SLIGHT BLUR
# ------------------------------------------------------------

base_blend = cv2.GaussianBlur(
    base_blend,
    (0,0),
    0.6
)

# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

blended = np.clip(
    base_blend,
    0,
    1
)


# ============================================================
# TEXTURE MODULATION
# ============================================================

blurred_dst = cv2.GaussianBlur(
    dst_f,
    (0,0),
    5
)

texture = dst_f - blurred_dst

blended += (
    TEXTURE_STRENGTH
    * texture
    * alpha_3
)

blended = np.clip(
    blended,
    0,
    1
)


# ============================================================
# FINAL OUTPUT
# ============================================================

result = (
    blended * 255
).astype(np.uint8)

cv2.imwrite(
    "final_result.png",
    result
)

print("Saved final_result.png")

cv2.imshow(
    "Final Result",
    result
)

cv2.waitKey(0)

cv2.destroyAllWindows()