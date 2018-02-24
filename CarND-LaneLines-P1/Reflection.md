# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/laneLines_thirdPass.jpg "LaneLines"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applyed Gaussian smoothing to blur the grayscaled image, then I use Canny edge detection to detect edges from the blured image, then I defined a four sided polygon to mask the image to get region-of-interest, finally I run hough tranform algorithm on the edge detected image to get candidate lane lines.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by filtering the candidate lane lines and applying average/extrapolate the line segments. I calculate the slope of the label it as left lane if the slope is between 30 to 60 degree, or label it as right lane if the slope is between -30 to -60 degree. Then I averaged the slopes and y-intercepts of the lanes, and use the averages to extropolate the line to the bottom of the image.

The final detected lane lines should be like this: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the light changes or shadow mixed with the lane lines, a lot of faked line candidtes would be detected. The detection algorithm can be very unstable.

Another shortcoming could be the current algorithm cannot detect curve lane lines. Thus, when we approach a bend in path, the algorithm will fail to detect it.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to smooth the lane lines by time. i.e. use the former lane line data to interpret and stablize the future lane lines.

Another potential improvement could be to adapt the parameters according to the brightness of the image. So that the result can be more robust.
