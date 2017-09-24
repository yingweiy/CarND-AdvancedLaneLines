**Udacity CarND Term 1 Project 4**
## Advanced Lane Finding Project
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./CamUndistorResult.png "Undistorted"
[test1]: ./test_images/test1.jpg "Road Transformed"
[test1_undist]: ./output_images/test1_undist.jpg ""
[test1_th]: ./output_images/test1_th.jpg ""
[test1_roi]: ./output_images/test1_roi.jpg ""
[test1_lines]: ./output_images/test1_lines.jpg ""
[test1_warped]: ./output_images/test1_warped.jpg ""
[test1_final]: ./output_images/test1_final.jpg ""

[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[perspective]: ./output_images/perspectiveTransform.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Camera matrix and distortion coefficients are computed by the function "CamCalibration()" in CamCalibration.py line 7-35.

I start by preparing "object points", which will be the (x, y, z) coordinates of the 
chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) 
plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended 
with a copy of it every time I successfully detect all chessboard corners in a test image.  
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in 
the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration 
and distortion coefficients using the `cv2.calibrateCamera()` function. The matrices are saved in 
a pickle file named "cam_calib.p" for future use.  

The image undistortion is implemented in the function corners_unwarp() in CamUndistort.py, I applied this distortion correction to the test image using 
the `cv2.undistort()` function and followed by `cv2.warpPerspective()`. The result is shown below: 

![alt text][image1]

### Pipeline (single images)

The pipeline code is in P4.ipynb jupyter notebook file.

#### 1. Provide an example of a distortion-corrected image.

This step is implemented in the cell under "1. Camera Undistort" section in the jupyter notebook.

To demonstrate this step, I will describe how I apply the distortion correction to one of
 the test images like this one:
![alt text][test1]

the undistor image looks like below:

![alt text][test1_undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

This step is implemented in the cell under "2. Color and gradient thresholding" section
 in the jupyter notebook.

Here's an example of my output for this step. I used a combination of color and gradient thresholds to generate a pseudo-colored image:
* Red channel: the edge channel by Sobel filter.
* Green channel: the saturation channel filtering.
* Blue channel: the hue channel filtering results.
 
![alt text][test1_th]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

This step is implemented in the cell under "4. Perspect Transform" section
 in the jupyter notebook.

The code for my perspective transform includes a function called `PerspectiveTransform()`, 
The function takes as inputs an image (`img`). 
I chose the hardcode the source and destination points in the following manner:

```python
y, x = img.shape[0], img.shape[1]    
hx1, hx2 = 548, 736
hy = 461   #the horizon
offset = 200
src = np.float32([[hx1, hy], [hx2, hy], [0, y-1], [x-1, y-1]])    
dst = np.float32([[offset, 0], [x-1-offset, 0], [offset, y-1], [x-1-offset, y-1]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 548, 461      | 200, 0        | 
| 736, 461      | 1079, 0      |
| 0, 719        | 200, 719      |
| 1279, 719      | 1079, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][perspective]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


This step is implemented in the cell under "5. Tracking Lanes" section
 in the jupyter notebook.
 
The main function of this part is the ``TrackingLanes()`` function. It contains the following steps:
* Calculating the histogram of the pixels in the image, and smooth it using a gaussian filter, as defined
in function ``FindPeakInSegment()``. The function then return the left peak position (lp) and right 
peak location (rp) for the whole lane lines.
*  Estimating the width of the lane in pixel by width = rp-lp
* Apply a point selection to get the left lane points in an array LPA and right lane line points 
RPA. This is implemented in function ``LanePointsSelection()``
* Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][test1_lines]

and for this particular example, the perspective transformed result is:

![alt text][test1_warped]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This step is implemented in the cell under "5. Tracking Lanes" section
 in the jupyter notebook, in the function ``TrackingLanes()``.  

```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/width # meters per pixel in x dimension

left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)   
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
``` 

Note that the ``width`` variable
 below is calculated width from previous step for a more accurate estimation of curvature.
    
```python
lp, rp, hist = FindPeakInSegment(gray)
width = rp - lp
```
        
The final reported curvature value in the video is the average of the left and right curvature radius.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
This step is implemented in the cell under "7. Lane Visualization" section
 in the jupyter notebook, in the function ``DrawingLaneCarpet()``.

![alt text][test1_final]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

### Challenge 

Here's a [link to my challenge video result](./challenge_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##### What works well
* All the steps following the instructions work well. In addition, I found using the Gaussian 
smoothing on the pixel histogram is useful to get a very stable peak locations as in the function ``FindPeakInSegment()``,
also the interplation/smoothing from previous frame also useful.
* In the challenge video, I added a color equilization function as below. I found this is very helpful
preprocesing step to normalize the image.
```python
def equalizeHistColor(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    image_out = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return image_out
```

##### Limitations
* The parameters are very hard-coded. For example, in the first challenge video, I used a different set 
of parameters. This can be further improved. 
* The left lane line in the last a few frames in the challenge video, especially on the top part seems
not correct (too much bended to the right). It is due to the color of "yellow" line is very blurry.


