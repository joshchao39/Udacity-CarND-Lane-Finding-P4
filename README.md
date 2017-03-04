# Advanced Lane Finding Project


[//]: # (Image References)
[image1]: ./output_images/undistorted_checker.jpg
[image2]: ./output_images/undistorted.jpg
[image3]: ./output_images/thresholded_image.png
[image4]: ./output_images/warped_with_lines.png
[image5]: ./output_images/warped_binary_with_lines.png
[image6]: ./output_images/fit_lines.png
[image7]: ./output_images/augmented_image.png

###Project Strcture
* `lane_finder.py` contains all essential code to generate the images and video  
* `utils.py` contains basic utility functions
* `test_code.py` contains code to troubleshoot/experiment the system
* `README.md` summarizes the steps and the results

###Camera Calibration
#####Goal
Remove the distortion in the image caused by the camera so we have the most **geometrically accurate** picture from the real world.

#####Method
20 chessboard pattern calibration images were provided, and `cv2.findChessboardCorners()` were able to find the pattern on 17 of them, which is used to generate the calibration matrices.

The code can be found in method `get_calibration_matrices()`

Since the calibration takes a few seconds and the results are deterministic, I pickled the matrices as `distortion.pickle` to be reused for later part of the project.

#####Example
This is the what one calibration image looks like before and after the distortion is removed:
 
![alt text][image1]

This is a road image before and after the un-distortion:

![alt text][image2]

###Lane Extraction
#####Goal
Identify where the lanes are by converting the color image into a binary image with lane clearly highlighted.
Note it is okay to have noise in other part of the image as we will mask in only the area around the lane by perspective warping later.

#####Method
- Edge Detection

**Sobel operator** was used in both x and y direction and calculate the magnitude and direction of the gradient vector. This allows me to filter out weak edges or horizontal lines.

- Color Channel Filter

I converted the color space to **HLS** and pick the pixels with high saturation (S channel), which is one of the consistent characteristics of the lane. I also exclude pixels with too little lightness (L channel) as the saturation channel becomes less stable when there are not enough pixel values.

- Combined

Edge detection provides precise lane detection when there is good contrast between the road and the lane, and saturation works well even when the road has similar brightness as the lane. The two methods complement each other.

The code can be found in `create_binary_image()` method.

#####Example

![alt text][image3]

###Perspective Transform
#####Goal

Convert the perspective of viewing the image from being in the car to being directly above the road (**birds-eye view**). This allows us to fit quadratic lines without having to worry about geometry change due to distance.

#####Method

I hand-picked the source and destination points by measuring on an image with straight lanes. Some heuristics can also be used because we know the camera is in the center of the car so the points have to be symmetrical. 

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 673, 443      | 920, 0        | 
| 1062, 705     | 920, 720      |
| 218, 705      | 360, 720      |
| 607, 443      | 360, 0        |

With these two sets of points we can calculate the transformation matrix, as well as the inverse transformation matrix to warp back any augmentation later by doing:
```
# Calculate the transformation matrix and inverse transformation matrix
M = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
Minv = cv2.getPerspectiveTransform(DST_PTS, SRC_PTS)
# Transform an image
warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
```

#####Example
To better illustrate the process the source and destination points are drawn on the image before and after the transformation.

![alt text][image4]

Just like the color image, we can transform the binary image as well. This gets us one step closer to identifying the lanes.

![alt text][image5]

###Lane Fitting 
#####Goal
Establish a function to describe both lanes from the detected pixels in the warped space.

#####Method
1. Sliding Window

Since the warped binary image still contains some noise and we are fitting two lines in the same image, we need to **trace** the lines. It was done by a sliding window approach where the search region in each window depends on the previous window.

To start, we find the two x locations in center portion of the bottom third of the image with the highest concentration of the detected pixels. This is our first window.
 
The center of the next window will be the mean x values of the detected pixels in the previous window, and we go on until we have all the windows positioned in the image.

We include only the pixels in the window to fit the quadratic lines, for both left lane and right lane separately.

This is done in `do_sliding_window()` method.

2. Adjust from Previous Frame

Once we have the quadratic lines, we can simply search around the fitted lines in the next frame. This yields higher precision and speed as we already know approximately where the lines should be.

This is done in `adjust_coefficients()` method.

#####Example
This is the sliding window example. Note the green boxes are the windows, the red pixels are included for the left lane, and the blue pixels are included for the right lane. The yellow lines are the fitted quadratic lines.

![alt text][image6]

Note the sliding window method doesn't have to be 100% precise, as the lines will adjust itself as more frames are processed in the video.

###Augmentation
#####Goal
Display all the information we have gathered back to the user, including where the lanes are, radius of the curvature, and position of the car in the lane.

#####Method

With the two lines we can:

1. Create an overlay to show where the detected lanes are. (Use `cv2.fillPoly()`)

2. Compute the radius of the curvature the car is going through. (More detail of the calculation [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php))

3. Compute the position of the car in the lane. (How far away are the lines from the center of the image)

The fitted lines has pixel as the unit, and we want the radius and position in meters, so I picked the conversion based on the fact that lane width is 3.7m in the US and the distance to the top of the perspective warp region is about 50 meters away.

```
M_PER_PIX_Y = 50 / 720  # meters per pixel in y dimension
M_PER_PIX_X = 3.7 / (920 - 360)  # meters per pixel in x dimension
```

#####Example

![alt text][image7]

###Video
The final product!
<p align="center">
  <img src="./output_images/augmented1.gif" alt="Augmented Driving Video"/>
</p>

Here's the [full video](https://youtu.be/Gto2y9o7MqI)

---

###Discussion

The hardest part of the project is finding a reliable way to extract the lanes from a color image. Human leverages a lot of other information to "zoom in" on what the lanes might look like, based on past experience, such as geometry and weather condition, but computer has to start from scratch for every frame. Perhaps machine learning can help here. It certainly makes one appreciate how well human is capable of on vision task.