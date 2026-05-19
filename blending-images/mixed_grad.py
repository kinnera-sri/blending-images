import cv2
import numpy as np

clicked_pts = []


def get_points(event, x, y, flags, param):

    global clicked_pts

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(clicked_pts) >= 4:
            return

        clicked_pts.append((x, y))

        cv2.circle(
            param,
            (x, y),
            6,
            (0,255,0),
            -1
        )

        cv2.imshow(
            "Select 4 Points",
            param
        )


def interactive_paste(bg_path, logo_path):

    global clicked_pts

    clicked_pts.clear()

    bg = cv2.imread(bg_path)

    logo = cv2.imread(
        logo_path,
        cv2.IMREAD_UNCHANGED
    )

    if bg is None or logo is None:
        print("Could not load images")
        return


    # --------------------------------------------------------
    # HANDLE ALPHA PNG
    # --------------------------------------------------------

    if logo.shape[2] == 4:

        alpha = logo[:,:,3]

        logo = logo[:,:,:3]

    else:

        alpha = np.ones(
            logo.shape[:2],
            dtype=np.uint8
        ) * 255


    # --------------------------------------------------------
    # SELECT POINTS
    # --------------------------------------------------------

    clone = bg.copy()

    cv2.namedWindow("Select 4 Points")

    cv2.setMouseCallback(
        "Select 4 Points",
        get_points,
        clone
    )

    print(
        "Click:\n"
        "Top-left\n"
        "Top-right\n"
        "Bottom-right\n"
        "Bottom-left"
    )

    while len(clicked_pts) < 4:

        cv2.imshow(
            "Select 4 Points",
            clone
        )

        key = cv2.waitKey(20) & 0xFF

        if key == 27:

            cv2.destroyAllWindows()

            return

    cv2.destroyAllWindows()


    # --------------------------------------------------------
    # HOMOGRAPHY
    # --------------------------------------------------------

    h, w = logo.shape[:2]

    pts_src = np.float32([
        [0,0],
        [w-1,0],
        [w-1,h-1],
        [0,h-1]
    ])

    pts_dst = np.float32(clicked_pts)

    H = cv2.getPerspectiveTransform(
        pts_src,
        pts_dst
    )

    warped_logo = cv2.warpPerspective(
        logo,
        H,
        (bg.shape[1], bg.shape[0])
    )

    warped_alpha = cv2.warpPerspective(
        alpha,
        H,
        (bg.shape[1], bg.shape[0])
    )


    # --------------------------------------------------------
    # CLEAN MASK
    # --------------------------------------------------------

    mask = cv2.GaussianBlur(
        warped_alpha,
        (11,11),
        0
    )

    # optional brightness tweak
    warped_logo = cv2.convertScaleAbs(
        warped_logo,
        alpha=1.1,
        beta=10
    )


    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    r = cv2.boundingRect(
        pts_dst.astype(np.int32)
    )

    center = (
        r[0] + r[2] // 2,
        r[1] + r[3] // 2
    )


    # --------------------------------------------------------
    # SEAMLESS CLONE
    # --------------------------------------------------------

    try:

        result = cv2.seamlessClone(
            warped_logo,
            bg,
            mask,
            center,
            cv2.MIXED_CLONE
        )

        # result = cv2.seamlessClone(
        #     warped_logo,
        #     bg,
        #     mask,
        #     center,
        #     cv2.MIXED_CLONE
        # )


        # --------------------------------------------------------
        # SAVE OUTPUT
        # --------------------------------------------------------
        
        output_path = "final_result_mixed_grad.png"
        
        cv2.imwrite(
            output_path,
            result
        )
        
        print(f"\nSaved result to: {output_path}")
            
        cv2.imshow(
                "Result",
                result
            )
    
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    except cv2.error as e:

        print("\nSeamless clone failed")
        print(e)


interactive_paste(
    "blending-images/main_images/b99_2.png",
    "blending-images/main_images/lego.png"
)