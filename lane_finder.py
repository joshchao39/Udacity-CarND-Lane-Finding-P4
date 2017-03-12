import time

from moviepy.editor import VideoFileClip

from image_operation import *

if __name__ == "__main__":
    start_time = time.time()

    mtx, dist = get_calibration_matrices()
    M = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
    Minv = cv2.getPerspectiveTransform(DST_PTS, SRC_PTS)

    LANE_DATA.mtx = mtx
    LANE_DATA.dist = dist
    LANE_DATA.M = M
    LANE_DATA.Minv = Minv

    """Compute the camera calibration matrix and distortion coefficients given a set of chessboard images."""
    if False:  # Output calibration image
        calib_img = mpimg.imread('camera_cal/calibration1.jpg')
        undist = cv2.undistort(calib_img, mtx, dist, None, mtx)
        save_2_images(calib_img, undist, OUT_IMG_DIR + 'Original Image', 'Undistorted Image', 'undistorted_checker.jpg')

    """Apply a distortion correction to raw images."""
    if False:  # Output undistorted image
        img = mpimg.imread(TEST_IMG_DIR + 'straight_lines1.jpg')
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        save_2_images(img, undist, 'Original Image', 'Undistorted Image', OUT_IMG_DIR + 'undistorted.jpg')

    """Use color transforms, gradients, etc., to create a thresholded binary image."""
    if False:  # Output thresholded binary image
        img = mpimg.imread(TEST_IMG_DIR + 'test5.jpg')
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        bin_img = create_binary_image(undist)
        save_2_images(img, bin_img, 'Original Image', 'Thresholded Binary Image',
                      OUT_IMG_DIR + 'thresholded_image.png', cmap2='gray')

    """Apply a perspective transform to rectify binary image ("birds-eye view")."""
    if False:  # Output perspective warped images with source/destination points drawn
        img = mpimg.imread(TEST_IMG_DIR + 'straight_lines1.jpg')
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        warped = cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)
        undist_lines, warped_lines = draw_perspective(undist, warped)
        # Create color warped image
        save_2_images(undist_lines, warped_lines, 'Undistorted Image', 'Warped Image',
                      OUT_IMG_DIR + 'warped_with_lines.png')

        # Create thresholded binary image
        bin_img = create_binary_image(undist)
        # Convert binary image to 3 channel image for line color
        bin_img = np.uint8(np.dstack((bin_img, bin_img, bin_img)) * 255)
        # Warp the binary image
        bin_warped = cv2.warpPerspective(bin_img, M, (bin_img.shape[1], bin_img.shape[0]), flags=cv2.INTER_LINEAR)
        # Draw perspective lines
        undist_lines, warped_bin_lines = draw_perspective(undist, bin_warped)
        # Output to file
        save_2_images(undist_lines, warped_bin_lines, 'Undistorted Image', 'Warped Binary Image',
                      OUT_IMG_DIR + 'warped_binary_with_lines.png')

    """
    Detect lane pixels and fit to find the lane boundary.
    Determine the curvature of the lane and vehicle position with respect to center.
    Warp the detected lane boundaries back onto the original image.
    """
    if False:
        img = mpimg.imread(TEST_IMG_DIR + 'test2.jpg')
        # Un-distort the image
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # Threshold to create binary image
        bin_img = create_binary_image(undist)
        # Warp the binary image
        bin_warped = cv2.warpPerspective(bin_img, M, (bin_img.shape[1], bin_img.shape[0]), flags=cv2.INTER_LINEAR)
        # Create an output image to draw on and to visualize the result
        img_to_draw = (np.dstack((bin_warped, bin_warped, bin_warped)) * 255).astype(np.uint8)
        # Do sliding window search
        left_y, left_x, right_y, right_x = do_sliding_window(bin_warped, img_to_draw)
        # Overlay the original image with augmentation
        lane_drawn = get_augment_overlay(bin_warped, LANE_DATA.left_fit_coef, LANE_DATA.right_fit_coef, left_y, left_x,
                                         right_y, right_x)
        augmented_undist = cv2.addWeighted(undist, 1, lane_drawn, 1, 0)
        # Output to files
        save_2_images(bin_img, img_to_draw, 'Binary Image', 'Warped Binary Image',
                      OUT_IMG_DIR + 'fit_lines.png', cmap1='gray')
        save_2_images(undist, augmented_undist, 'Undistorted Image', 'Augmented Image',
                      OUT_IMG_DIR + 'augmented_image.png')

    """Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position."""
    if True:
        clip = VideoFileClip("project_video.mp4")
        processed_clip = clip.fl_image(detect_lane)  # NOTE: this function expects color images!!
        processed_clip.write_videofile(OUT_IMG_DIR + 'project_video_augmented.mp4', audio=False)

    print("--- %s seconds ---" % (time.time() - start_time))
