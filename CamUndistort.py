import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "cam_calib.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('camera_cal/calibration3.jpg')

nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE

def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # 2) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # 4) If corners found:
    if ret == True:
            # a) draw corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code


            #src = np.float32([[, ], [, ], [, ], [, ]])
            src = np.float32([corners[0,:], corners[nx - 1,:],
                              corners[nx * ny - nx,:], corners[nx * ny - 1,:]])

            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            # bs = 150
            #dst = np.float32([[bs/2, bs/2], [(nx-0.5)*bs, bs/2],
            #                  [bs/2, (ny-0.5)*bs], [(nx-0.5)*bs, (ny-0.5)*bs]])

            offset=100
            img_size = (gray.shape[1], gray.shape[0])
            dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                              [offset, img_size[1] - offset],
                              [img_size[0] - offset, img_size[1] - offset]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix

            M = cv2.getPerspectiveTransform(src, dst)

            # e) use cv2.warpPerspective() to warp your image to a top-down view

            warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    else:
        warped = None
        M = None
        print('Error: corners not found.')

    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
if not(top_down is None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(top_down)
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
else:
    print('Unable to distort.')