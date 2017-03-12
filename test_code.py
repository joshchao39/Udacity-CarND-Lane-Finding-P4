from lane_finder import *
import numpy as np

TEST_OUTPUT_DIR = 'test_output/'

"""Common"""
start_time = time.time()
test_paths = [TEST_IMG_DIR + f for f in os.listdir(TEST_IMG_DIR)]

"""Calibration"""
if False:
    # Recalculaate calibration matrices
    get_calibration_matrices(True)

"""Threshold"""
if False:
    for path in test_paths:
        img = mpimg.imread(path)
        # Undistort the image
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # Threshold to create binary image
        bin_output = create_binary_image(undist)
        # Save outputs
        save_2_images(img, bin_output, 'Original Image', 'Binary Image',
                      TEST_OUTPUT_DIR + os.path.split(path)[-1], cmap2='gray')

"""Warp perspective (Color)"""
if False:
    for path in test_paths:
        img = mpimg.imread(path)
        # Undistort the image
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # Warp the undistorted image
        warped = cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)
        # Draw lines
        undist_lines, warped_lines = draw_perspective(undist, warped)
        # Save outputs
        save_2_images(undist_lines, warped_lines, 'Undistorted Image', 'Warped Image',
                      TEST_OUTPUT_DIR + os.path.split(path)[-1])

"""Warp perspective (Binary)"""
if False:
    for path in test_paths:
        img = mpimg.imread(path)
        # Undistort the image
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # Threshold to create binary image
        bin_img = create_binary_image(undist)
        # Convert binary image to 3 channel image
        bin_img = np.uint8(np.dstack((bin_img, bin_img, bin_img)) * 255)
        # Warp the undistorted image
        warped_bin = cv2.warpPerspective(bin_img, M, (bin_img.shape[1], bin_img.shape[0]), flags=cv2.INTER_LINEAR)
        # Draw lines
        undist_lines, warped_bin_lines = draw_perspective(undist, warped_bin)
        # Save outputs
        save_2_images(undist_lines, warped_bin_lines, 'Undistorted Image', 'Warped Binary Image',
                      TEST_OUTPUT_DIR + os.path.split(path)[-1])

"""Fit lanes"""
if True:
    for path in test_paths:
        if 'test6' not in path:
            continue
        print("path", path)
        LANE_DATA.reset()
        img = mpimg.imread(path)
        # Un-distort the image
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # Threshold to create binary image
        bin_img = create_binary_image(undist)
        # Warp the undistorted image
        bin_warped = cv2.warpPerspective(bin_img, M, (bin_img.shape[1], bin_img.shape[0]), flags=cv2.INTER_LINEAR)
        # Create an output image to draw on and to visualize the result
        img_to_draw = (np.dstack((bin_warped, bin_warped, bin_warped)) * 255).astype(np.uint8)
        # Do sliding window search
        left_y, left_x, right_y, right_x = do_sliding_window(bin_warped, img_to_draw)
        # Overlay the original image with augmentation
        lane_drawn = get_augment_overlay(bin_warped, LANE_DATA.left_fit_coef, LANE_DATA.right_fit_coef, left_y, left_x,
                                         right_y, right_x)
        augmented_undist = cv2.addWeighted(undist, 1, lane_drawn, 1, 0)

        # save_2_images(undist, img_to_draw, 'Undistorted Image', 'Warped Binary Image',
        #               TEST_OUTPUT_DIR + os.path.split(path)[-1], cmap1='gray')
        save_2_images(undist, augmented_undist, 'Undistorted Image', 'Augmented Image',
                      TEST_OUTPUT_DIR + os.path.split(path)[-1])

"""Random code snippets"""
# cv2.imshow('Image with Corners', img_corners)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))
