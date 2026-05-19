import cv2 as cv

def merge(foreground_path, background_path, alpha, beta):

    img = cv.imread(foreground_path)
    background = cv.imread(background_path)

    background = cv.resize(background, (img.shape[1], img.shape[0]))

    final = cv.addWeighted(img, alpha, background, beta, 0)

    # cv.imshow('Foreground Image', img)
    cv.imshow('Merged Image', final)
    cv.waitKey(0)
    cv.destroyAllWindows()

bg = "/home/kinnera/Documents/gyrus_kinnera/blurring/pepsi_logo.png"
fg = "/home/kinnera/Documents/gyrus_kinnera/blurring/apple.png"
merge(fg, bg, 0.75, 0.25)