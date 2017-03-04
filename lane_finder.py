import math
import os
import pickle
import time

import cv2
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from utils import *

OUT_IMG_DIR = 'output_images/'
TEST_IMG_DIR = 'test_images/'

# Color
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

# Calibration
CALIBRATION_DIR = "camera_cal/"
CALIB_CORNER_CNT_X = 9
CALIB_CORNER_CNT_Y = 6

# Threshold
SOBEL_K = 3
SOBEL_DIR_THR_L = 0.5 * math.pi / 2
SOBEL_DIR_THR_H = 0.9 * math.pi / 2
SOBEL_MAG_THR_L = 40
SOBEL_MAG_THR_H = 255
S_THR_L = 100
S_THR_H = 255
L_THR_L = 0
L_THR_H = 40

# Warp perspective
SRC_PTS = np.float32(
    [[673, 443],
     [1062, 705],
     [218, 705],
     [607, 443]])

DST_PTS = np.float32(
    [[920, 0],
     [920, 720],
     [360, 720],
     [360, 0]])

M_PER_PIX_Y = 50 / 720  # meters per pixel in y dimension
M_PER_PIX_X = 3.7 / (920 - 360)  # meters per pixel in x dimension

# Curvature Fitting
HISTOGRAM_SIDE_SKIP = 100  # Side of image to skip when calculating the histogram to avoid including the next lane
NUM_WINDOWS = 9  # Number of sliding windows
WIN_MARGIN = 110  # Width of the windows +/- margin
MIN_PIX = 120  # Minimum number of pixels found to recenter window
LEARNING_RATE = 0.5  # How much of the new coefficients to use
LIVE_MARGIN = 50  # How wide to look for valid pixels from the last fitted line


class LaneData:
    """Object to store lane data between each frame"""

    def __init__(self):
        self.left_fit_coef = None
        self.right_fit_coef = None
        self.radius = None
        self.pos = None

    def reset(self):
        self.left_fit_coef = None
        self.right_fit_coef = None
        self.radius = None
        self.pos = None

    def is_initialized(self):
        return self.left_fit_coef is not None and self.right_fit_coef is not None

    def update(self, new_left_fit_coef, new_right_fit_coef, new_radius, new_pos):
        if self.left_fit_coef is None:
            self.left_fit_coef = new_left_fit_coef
        else:
            a = LEARNING_RATE * new_left_fit_coef[0] + (1 - LEARNING_RATE) * self.left_fit_coef[0]
            b = LEARNING_RATE * new_left_fit_coef[1] + (1 - LEARNING_RATE) * self.left_fit_coef[1]
            c = LEARNING_RATE * new_left_fit_coef[2] + (1 - LEARNING_RATE) * self.left_fit_coef[2]
            self.left_fit_coef = [a, b, c]

        if self.right_fit_coef is None:
            self.right_fit_coef = new_right_fit_coef
        else:
            a = LEARNING_RATE * new_right_fit_coef[0] + (1 - LEARNING_RATE) * self.right_fit_coef[0]
            b = LEARNING_RATE * new_right_fit_coef[1] + (1 - LEARNING_RATE) * self.right_fit_coef[1]
            c = LEARNING_RATE * new_right_fit_coef[2] + (1 - LEARNING_RATE) * self.right_fit_coef[2]
            self.right_fit_coef = [a, b, c]

        if self.radius is None:
            self.radius = new_radius
        else:
            self.radius = LEARNING_RATE * new_radius + (1 - LEARNING_RATE) * self.radius

        if self.pos is None:
            self.pos = new_pos
        else:
            self.pos = LEARNING_RATE * new_pos + (1 - LEARNING_RATE) * self.pos


def get_calibration_matrices(recalculate=False):
    if recalculate:
        calib_paths = [CALIBRATION_DIR + f for f in os.listdir(CALIBRATION_DIR)]
        objp = np.zeros((CALIB_CORNER_CNT_Y * CALIB_CORNER_CNT_X, 3), np.float32)
        objp[:, :2] = np.mgrid[0:CALIB_CORNER_CNT_X, 0:CALIB_CORNER_CNT_Y].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []
        test_images = []
        for path in calib_paths:
            # reading in an image
            img = mpimg.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (CALIB_CORNER_CNT_X, CALIB_CORNER_CNT_Y), None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                test_images.append(img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, test_images[0].shape[0:2], None, None)

        with open('distortion.pickle', 'wb') as mat_out:
            dist_mat = {'mtx': mtx, 'dist': dist}
            pickle.dump(dist_mat, mat_out)
    else:
        with open('distortion.pickle', 'rb') as mat_in:
            dist_mat = pickle.load(mat_in)
            mtx = dist_mat['mtx']
            dist = dist_mat['dist']
    return mtx, dist


def create_binary_image(img):
    # Apply Sobel operator
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=SOBEL_K)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=SOBEL_K)
    sobelx_abs = np.abs(sobelx)
    sobely_abs = np.abs(sobely)

    # Convert Sobel to magnitude and direction
    arctan_sobel = np.arctan2(sobely_abs, sobelx_abs)
    mag_sobel_raw = np.sqrt(np.square(sobelx) + np.square(sobely))
    mag_sobel = np.uint8(255 * mag_sobel_raw / np.max(mag_sobel_raw))

    # Sobel Threshold
    bin_sobel = np.zeros_like(arctan_sobel)
    bin_sobel[(SOBEL_DIR_THR_L <= arctan_sobel) & (arctan_sobel <= SOBEL_DIR_THR_H) & (
        SOBEL_MAG_THR_L <= mag_sobel) & (mag_sobel <= SOBEL_MAG_THR_H)] = 1

    # Convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    # HLS Threshold
    bin_hls = np.zeros_like(s)
    bin_hls[(S_THR_L < s) & (s < S_THR_H)] = 1
    bin_hls[(L_THR_L < l) & (l < L_THR_H)] = 0

    # Combine binary images
    binary_image = np.zeros_like(bin_sobel)
    binary_image[(bin_sobel == 1) | (bin_hls == 1)] = 1

    return binary_image


def draw_perspective(img, warped):
    img_drawn = img.copy()
    cv2.line(img_drawn, (int(SRC_PTS[0, 0]), int(SRC_PTS[0, 1])), (int(SRC_PTS[1, 0]), int(SRC_PTS[1, 1])),
             color=RED, thickness=3)
    cv2.line(img_drawn, (int(SRC_PTS[1, 0]), int(SRC_PTS[1, 1])), (int(SRC_PTS[2, 0]), int(SRC_PTS[2, 1])),
             color=RED, thickness=3)
    cv2.line(img_drawn, (int(SRC_PTS[2, 0]), int(SRC_PTS[2, 1])), (int(SRC_PTS[3, 0]), int(SRC_PTS[3, 1])),
             color=RED, thickness=3)
    cv2.line(img_drawn, (int(SRC_PTS[3, 0]), int(SRC_PTS[3, 1])), (int(SRC_PTS[0, 0]), int(SRC_PTS[0, 1])),
             color=RED, thickness=3)

    warped_drawn = warped.copy()
    cv2.line(warped_drawn, (int(DST_PTS[0, 0]), int(DST_PTS[0, 1])), (int(DST_PTS[1, 0]), int(DST_PTS[1, 1])),
             color=RED, thickness=3)
    cv2.line(warped_drawn, (int(DST_PTS[1, 0]), int(DST_PTS[1, 1])), (int(DST_PTS[2, 0]), int(DST_PTS[2, 1])),
             color=RED, thickness=3)
    cv2.line(warped_drawn, (int(DST_PTS[2, 0]), int(DST_PTS[2, 1])), (int(DST_PTS[3, 0]), int(DST_PTS[3, 1])),
             color=RED, thickness=3)
    cv2.line(warped_drawn, (int(DST_PTS[3, 0]), int(DST_PTS[3, 1])), (int(DST_PTS[0, 0]), int(DST_PTS[0, 1])),
             color=RED, thickness=3)

    return img_drawn, warped_drawn


def get_start_location(binary_warped_img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped_img[binary_warped_img.shape[0] * 2 / 3:, HISTOGRAM_SIDE_SKIP:-HISTOGRAM_SIDE_SKIP],
                       axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    left_x_start = HISTOGRAM_SIDE_SKIP + np.argmax(histogram[:midpoint])
    right_x_start = HISTOGRAM_SIDE_SKIP + midpoint + np.argmax(histogram[midpoint:])
    return left_x_start, right_x_start


def generate_poly_values(shape, left_fit_coef, right_fit_coef):
    # Generate x and y values for plotting
    plot_y = np.linspace(0, shape[0] - 1, shape[0])
    plot_x_left = left_fit_coef[0] * np.square(plot_y) + left_fit_coef[1] * plot_y + left_fit_coef[2]
    plot_x_right = right_fit_coef[0] * np.square(plot_y) + right_fit_coef[1] * plot_y + right_fit_coef[2]
    return plot_y, plot_x_left, plot_x_right


def do_sliding_window(bin_warped, img_to_draw=None):
    # Find the x positions to start the trace
    left_x_start, right_x_start = get_start_location(bin_warped)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Set height of windows
    window_height = np.int(bin_warped.shape[0] / NUM_WINDOWS)

    # Current positions to be updated for each window
    left_x_current = left_x_start
    right_x_current = right_x_start

    # Create empty lists to receive left and right lane pixel indices
    left_lane_indices = []
    right_lane_indices = []

    # Step through the windows one by one
    for window in range(NUM_WINDOWS):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = bin_warped.shape[0] - (window + 1) * window_height
        win_y_high = bin_warped.shape[0] - window * window_height
        win_x_left_low = max(left_x_current - WIN_MARGIN, 0)
        win_x_left_high = min(left_x_current + WIN_MARGIN, bin_warped.shape[1])
        win_x_right_low = max(right_x_current - WIN_MARGIN, 0)
        win_x_right_high = min(right_x_current + WIN_MARGIN, bin_warped.shape[1])

        if img_to_draw is not None:
            # Draw the windows on the visualization image
            cv2.rectangle(img_to_draw, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), color=GREEN,
                          thickness=2)
            cv2.rectangle(img_to_draw, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), color=GREEN,
                          thickness=2)

        # Identify the nonzero pixels in x and y within the window
        left_indices_to_add = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_x_left_low) & (
            nonzero_x < win_x_left_high)).nonzero()[0]
        right_indices_to_add = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_x_right_low) & (
            nonzero_x < win_x_right_high)).nonzero()[0]

        # Append included indices to the lists
        left_lane_indices.extend(left_indices_to_add)
        right_lane_indices.extend(right_indices_to_add)

        # Recenter next window on their mean position (move left and right window in parallel)
        change = 0
        if len(left_indices_to_add) > MIN_PIX:
            change += np.int(np.mean(nonzero_x[left_indices_to_add])) - left_x_current
        if len(right_indices_to_add) > MIN_PIX:
            change += np.int(np.mean(nonzero_x[right_indices_to_add])) - right_x_current
        left_x_current += int(change / 2)
        right_x_current += int(change / 2)

    # Extract left and right line pixel positions
    left_x = nonzero_x[left_lane_indices]
    left_y = nonzero_y[left_lane_indices]
    right_x = nonzero_x[right_lane_indices]
    right_y = nonzero_y[right_lane_indices]

    # Fit a second order polynomial to each
    left_fit_coef = np.polyfit(left_y, left_x, 2)
    right_fit_coef = np.polyfit(right_y, right_x, 2)

    # Find lane curvature and position in lane at the vehicle
    curvature = calculate_turning_radius(bin_warped.shape[0], left_y, right_y, left_x, right_x)
    car_pos = calculate_position(bin_warped.shape[0], left_fit_coef, right_fit_coef, bin_warped.shape[1] / 2)

    if img_to_draw is not None:
        # Change included activations to red color
        img_to_draw[left_y, left_x] = RED
        img_to_draw[right_y, right_x] = BLUE
        # Use fillPoly to draw the fitted lines
        plot_y, plot_x_left, plot_x_right = generate_poly_values(bin_warped.shape, left_fit_coef, right_fit_coef)
        left_line_fit_window1 = np.array([np.transpose(np.vstack([plot_x_left - 1, plot_y]))])
        left_line_fit_window2 = np.array([np.flipud(np.transpose(np.vstack([plot_x_left + 1, plot_y])))])
        left_line_fit_pts = np.hstack((left_line_fit_window1, left_line_fit_window2))
        right_line_fit_window1 = np.array([np.transpose(np.vstack([plot_x_right - 1, plot_y]))])
        right_line_fit_window2 = np.array([np.flipud(np.transpose(np.vstack([plot_x_right + 1, plot_y])))])
        right_line_fit_pts = np.hstack((right_line_fit_window1, right_line_fit_window2))
        cv2.fillPoly(img_to_draw, np.int_([left_line_fit_pts]), color=YELLOW)
        cv2.fillPoly(img_to_draw, np.int_([right_line_fit_pts]), color=YELLOW)

    lane_data.update(left_fit_coef, right_fit_coef, curvature, car_pos)

    return left_y, left_x, right_y, right_x


def adjust_coefficients(bin_warped):
    left_coef = lane_data.left_fit_coef
    right_coef = lane_data.right_fit_coef

    nonzero = bin_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = (
        (nonzerox > (left_coef[0] * (nonzeroy ** 2) + left_coef[1] * nonzeroy + left_coef[2] - LIVE_MARGIN)) & (
            nonzerox < (left_coef[0] * (nonzeroy ** 2) + left_coef[1] * nonzeroy + left_coef[2] + LIVE_MARGIN)))
    right_lane_inds = (
        (nonzerox > (right_coef[0] * (nonzeroy ** 2) + right_coef[1] * nonzeroy + right_coef[2] - LIVE_MARGIN)) & (
            nonzerox < (right_coef[0] * (nonzeroy ** 2) + right_coef[1] * nonzeroy + right_coef[2] + LIVE_MARGIN)))

    # Again, extract left and right line pixel positions
    left_x = nonzerox[left_lane_inds]
    left_y = nonzeroy[left_lane_inds]
    right_x = nonzerox[right_lane_inds]
    right_y = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    new_left_coef = np.polyfit(left_y, left_x, 2)
    new_right_coef = np.polyfit(right_y, right_x, 2)

    # Find lane curvature and position in lane at the vehicle
    curvature = calculate_turning_radius(bin_warped.shape[0], left_y, right_y, left_x, right_x)
    car_pos = calculate_position(bin_warped.shape[0], new_left_coef, new_right_coef, bin_warped.shape[1] / 2)

    # Update stored coefficients
    lane_data.update(new_left_coef, new_right_coef, curvature, car_pos)

    return left_y, left_x, right_y, right_x


def get_augment_overlay(bin_warped, left_fit_coef, right_fit_coef, left_y, left_x, right_y, right_x):
    # Create blank image to draw
    warp_zero = np.zeros_like(bin_warped)
    drawn_lanes = np.dstack((warp_zero, warp_zero, warp_zero))

    # Draw the lanes onto the warped blank image
    plot_y, plot_x_left, plot_x_right = generate_poly_values(bin_warped.shape, left_fit_coef, right_fit_coef)
    pts_left = np.array([np.transpose(np.vstack([plot_x_left, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([plot_x_right, plot_y])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(drawn_lanes, np.int_([pts]), color=GREEN)

    # Draw the detected pixels onto warped blank image
    drawn_lanes[left_y, left_x] = RED
    drawn_lanes[right_y, right_x] = BLUE

    # Reduce overlay intensity
    drawn_lanes = (drawn_lanes * 0.3).astype(np.uint8)

    # Warp the overlap back to original image space using inverse perspective matrix
    overlay = cv2.warpPerspective(drawn_lanes, Minv, (bin_warped.shape[1], bin_warped.shape[0]))

    # Etch turning radius and position in lane on the overlay
    p_dir = 'left' if lane_data.pos >= 0 else 'right'
    radius_cap = 9999 if lane_data.radius >= 9999 else lane_data.radius
    text = 'Radius of Curvature:{:03.0f}m, Position:{:.2f}m {} of the center'.format(radius_cap, abs(lane_data.pos),
                                                                                     p_dir)
    cv2.putText(overlay, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=WHITE, thickness=2,
                lineType=cv2.LINE_AA)

    return overlay


def calculate_turning_radius(max_y, left_y, right_y, left_x, right_x):
    # Fit new polynomials to x,y in world space in meters
    left_fit_m = np.polyfit(left_y * M_PER_PIX_Y, left_x * M_PER_PIX_X, 2)
    right_fit_m = np.polyfit(right_y * M_PER_PIX_Y, right_x * M_PER_PIX_X, 2)
    # Calculate the new radii of curvature
    left_radius = ((1 + (2 * left_fit_m[0] * max_y * M_PER_PIX_Y + left_fit_m[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_m[0])
    right_radius = ((1 + (2 * right_fit_m[0] * max_y * M_PER_PIX_Y + right_fit_m[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_m[0])
    # Take average curvature between left and right
    radius = (left_radius + right_radius) / 2
    return radius


def calculate_position(max_y, left_coef, right_coef, midpoint):
    # Calculate the distance from the center of the lane
    left_pos = left_coef[0] * (max_y ** 2) + left_coef[1] * max_y + left_coef[2]
    right_pos = right_coef[0] * (max_y ** 2) + right_coef[1] * max_y + right_coef[2]
    middle_pos = (left_pos + right_pos) / 2
    car_pos = (middle_pos - midpoint) * M_PER_PIX_X
    return car_pos


def process_image(image):
    # Un-distort the image
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # Threshold to create binary image
    bin_img = create_binary_image(undist)

    # Warp the undistorted image
    bin_warped = cv2.warpPerspective(bin_img, M, (bin_img.shape[1], bin_img.shape[0]), flags=cv2.INTER_LINEAR)

    if not lane_data.is_initialized():
        # Do sliding window search in the first frame
        left_y, left_x, right_y, right_x = do_sliding_window(bin_warped)
    else:
        # Adjust the lane coefficients from last frame
        left_y, left_x, right_y, right_x = adjust_coefficients(bin_warped)

    # Overlay the original image with augmentation
    overlay = get_augment_overlay(bin_warped, lane_data.left_fit_coef, lane_data.right_fit_coef, left_y, left_x,
                                  right_y, right_x)
    augmented_undist = cv2.addWeighted(undist, 1, overlay, 1, 0)

    return augmented_undist


# Global variables (Yes I know it's bad!)
mtx, dist = get_calibration_matrices()
M = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
Minv = cv2.getPerspectiveTransform(DST_PTS, SRC_PTS)
lane_data = LaneData()

if __name__ == "__main__":
    start_time = time.time()

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
        lane_drawn = get_augment_overlay(bin_warped, lane_data.left_fit_coef, lane_data.right_fit_coef, left_y, left_x,
                                         right_y, right_x)
        augmented_undist = cv2.addWeighted(undist, 1, lane_drawn, 1, 0)
        # Output to files
        save_2_images(bin_img, img_to_draw, 'Binary Image', 'Warped Binary Image',
                      OUT_IMG_DIR + 'fit_lines.png', cmap1='gray')
        save_2_images(undist, augmented_undist, 'Undistorted Image', 'Augmented Image',
                      OUT_IMG_DIR + 'augmented_image.png')

    """Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position."""
    if True:
        # clip1 = VideoFileClip("project_video.mp4")
        # white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
        # white_clip.write_videofile(OUT_IMG_DIR + 'project_video_augmented.mp4', audio=False)
        clip1 = VideoFileClip("challenge_video.mp4")
        white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
        white_clip.write_videofile(OUT_IMG_DIR + 'challenge_video_augmented.mp4', audio=False)
        clip1 = VideoFileClip("harder_challenge_video.mp4")
        white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
        white_clip.write_videofile(OUT_IMG_DIR + 'harder_challenge_video_augmented.mp4', audio=False)

    print("--- %s seconds ---" % (time.time() - start_time))
