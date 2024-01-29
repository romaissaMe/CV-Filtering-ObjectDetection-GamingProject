# CV-Filtering-ObjectDetection-GamingProject

## Table of Contents

1. [Overview](#overview)
2. [Filters](#filters)
   - [Smoothing filters](#smoothing-filters)
     1. [Mean filter](#mean-filter)
     2. [Median filter](#median-filter)
   - [Convolution filters](#convolution-filters)
     1. [Gaussian filter](#gaussian-filter)
     2. [Laplacian filter](#laplacian-filter)
     3. [Sobel filter](#sobel-filter)
     4. [Emboss filter](#emboss-filter)
   - [Morphological filters](#morphological-filters)
     1. [Erosion filter](#erosion-filter)
     2. [Dilation filter](#dilation-filter)
3. [Object Detection by colour](#object-detection-by-colour)
4. [Invisiblity Cloak](#invisiblity-cloak)
5. [Green Screen](#green-screen)
6. [Gaming Project](#6-gaming-project)
7. [How to Run the Application](#7-how-to-run-the-application)

## 1. Overview <a name="overview"></a>

In this academic project, we meticulously developed custom algorithms from scratch to create a gaming project centered around object detection by color and filtering techniques. Our implementation includes the crafting of various filters and their application. Notably, we programmed our algorithms without relying on predefined functions. Additionally, we incorporated advanced features such as a green screen effect and an invisibility cloak effect, both carefully designed and implemented using our own algorithms

## 2. Filters <a name="filters"></a>

### 2.1. Smoothing filters <a name="smoothing-filters"></a>

#### 2.1.1 Mean filter <a name="mean-filter"></a>

The mean filter replaces the center value in the window with the average (mean) of all the pixel values in the window.

#### 2.1.2 Median filter <a name="median-filter"></a>

The median filter replaces the center value in the window with the median of all the pixel values in the window.

### 2.2. Convolution filters <a name="convolution-filters"></a>

#### 2.2.1 Gaussian filter <a name="gaussian-filter"></a>

The Gaussian filter is a smoothing filter that is used to blur images. It is a 2D convolution operator that is used to remove noise from an image. It is also used to reduce detail and decrease noise in an image.

#### 2.2.2 Laplacian filter <a name="laplacian-filter"></a>

The Laplacian filter is a second derivative edge detector. It computes the second derivatives of the image and then marks the points where the second derivative value is maximum.

#### 2.2.3 Sobel filter <a name="sobel-filter"></a>

The Sobel filter is a first derivative edge detector. It computes the first derivatives of the image separately for the X and Y axes. It is used to detect edges in an image.

#### 2.2.4 Emboss filter <a name="emboss-filter"></a>

The emboss filter is a filter that is used to highlight the edges in an image. It is a 3x3 convolution filter that is used to detect edges in an image.

### 2.3. Morphological filters <a name="morphological-filters"></a>

#### 2.3.1 Erosion filter <a name="erosion-filter"></a>

The erosion filter is used to erode away the boundaries of foreground objects in an image. It is used to remove small white noises from an image.

#### 2.3.2 Dilation filter <a name="dilation-filter"></a>

The dilation filter is used to increase the boundaries of foreground objects in an image. It is used to remove small black holes from an image.

## 3. Object Detection by colour <a name="object-detection-by-colour"></a>

In this part we implement a function `object_detection_by_colour` that detects an object by its colour. The function takes as input an image and a colour and returns the image with the detected object highlighted in white and the rest of the image in black. The function uses the HSV colour space to detect the object by its colour. The function uses the following steps:

1. Convert the image from RGB to HSV.
2. Create a mask that contains the pixels that have the same colour as the colour intervals of the input colour.
3. Apply the mask to the image.
4. Detect the largest contour in the image using Breadth First Search (BFS).
5. Draw the contour on the image.

## 4. Invisiblity Cloak <a name="invisiblity-cloak"></a>

In this part we implement a function `invisibility_cloak` that makes an object disappear from the image. The function takes as input an image and a colour and returns the image with the object with that colour removed. The function uses the following steps:

1. Save the background image.
2. Convert the image from RGB to HSV.
3. Create a mask that contains the pixels that have the same colour as the colour intervals of the input colour.
4. Apply the mask to the image.
5. Replace the pixels of the object with the pixels of the background image.

## 5. Green Screen <a name="green-screen"></a>

In this part we implement a function `green_screen` that replaces the pixels of a green screen with the pixels of a background image. The function takes as input an image and a background image and returns the image with the green screen replaced with the background image. The function uses the following steps:

1. Convert the image from RGB to HSV.
2. Create a mask that contains the pixels that have the same colour as the colour intervals of the green screen.
3. Apply the mask to the image.
4. Replace the pixels of the green screen with the pixels of the background image.

## 6. Gaming Project <a name="gaming-App"></a>
## 7. How to Run the Application

1. Install the required dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run app.py`
