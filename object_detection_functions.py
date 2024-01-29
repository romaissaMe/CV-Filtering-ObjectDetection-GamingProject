import numpy as np
import cv2 as cv
from collections import deque
import streamlit as st
from filters import filtreMedianeColored
from utilities import *

# ********************************************************************** Object Detection Functions ***********************************************


# NON optimized function
def detect_object_color_camera(
    cap,
    frame_placeholder,
    mask_placeholder,
    stop_btn,
    min_blue,
    min_green,
    min_red,
    max_blue,
    max_green,
    max_red,
):
    window_width = 800
    window_height = 600
    if not cap.isOpened():
        st.warning("Cannot open camera", icon="🚨")
        exit()
    while True and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            print("error image read")
            exit()
        img = cv.cvtColor(
            frame, cv.COLOR_BGR2HSV
        )  # convert image to HSV to facilitate object color detection
        frame_rgb = cv.cvtColor(
            frame, cv.COLOR_BGR2RGB
        )  # convert frame to RGB to display it
        h, w = img.shape[:2]
        # Create masks for the specified HSV ranges
        mask = generate_range_color_mask(
            h, w, img, min_blue, max_blue, min_green, max_green, min_red, max_red
        )  # genrate the mask based on colors range
        result_contours = find_contours_bfs(
            mask
        )  # generate contours of every object masked

        # Assuming largest means the contour with the most pixels
        if result_contours:
            largest_contour = max(result_contours, key=len)
            # Calculate the bounding box coordinates of the largest contour
            x_min = max(
                min(coord[1] for coord in largest_contour) - 3, 0
            )  # Ensure x_min is not negative
            y_min = max(
                min(coord[0] for coord in largest_contour) - 3, 0
            )  # Ensure y_min is not negative
            x_max = min(
                max(coord[1] for coord in largest_contour) + 3, img.shape[1] - 1
            )  # Ensure x_max is within image width
            y_max = min(
                max(coord[0] for coord in largest_contour) + 3, img.shape[0] - 1
            )  # Ensure y_max is within image height
            cv.rectangle(
                frame_rgb, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), 255, 2
            )  # draw a rectangle around the object detected by color

        # Display the mask and trackbars in the same window
        mask_placeholder.image(mask)
        # display the base img with contours
        frame_placeholder.image(frame_rgb)
        # Wait for ESC key (27) to exit the loop
        if cv.waitKey(1) == 27 or stop_btn:
            break

    cap.release()
    cv.destroyAllWindows()


# Optimized function
def detect_object_color_camera_optimized(
    cap,
    frame_placeholder,
    mask_placeholder,
    stop_btn,
    min_blue,
    min_green,
    min_red,
    max_blue,
    max_green,
    max_red,
):
    window_width = 800
    window_height = 600
    scale_factor = 0.1
    # cap = cv.VideoCapture(0)
    if not cap.isOpened():
        st.warning("Cannot open camera")
        exit()
    while True and not stop_btn:
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        frame_rgb = cv.cvtColor(
            frame, cv.COLOR_BGR2RGB
        )  # convert frame to RGB to display it
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv = resize(hsv, scale_factor)  # resize the hsv frame (optimization)
        # hsv = filtreMedianeColored(
        #     hsv, 3
        # )  # filter the hsv frame for better results (optimization)
        h, w = hsv.shape[:2]
        # Create masks for the specified HSV ranges
        mask = generate_range_color_mask(
            h, w, hsv, min_blue, max_blue, min_green, max_green, min_red, max_red
        )  # generate the mask based on range colors
        result_contours = find_contours_bfs(mask)  # define contours of masked objects

        if result_contours:
            largest_contour = max(result_contours, key=len)  # get the largest contour
            # the largest contour is calculated using the resized frame, so we need to rescale the largest contour to the original size
            largest_contour = [
                (int(y / scale_factor), int(x / scale_factor))
                for y, x in largest_contour
            ]
            # Draw a rectangle around the largest contour
            x_min = max(
                min(coord[1] for coord in largest_contour) - 3, 0
            )  # Ensure x_min is not negative
            y_min = max(
                min(coord[0] for coord in largest_contour) - 3, 0
            )  # Ensure y_min is not negative
            x_max = min(
                max(coord[1] for coord in largest_contour) + 3, frame.shape[1] - 1
            )  # Ensure x_max is within image width
            y_max = min(
                max(coord[0] for coord in largest_contour) + 3, frame.shape[0] - 1
            )  # Ensure y_max is within image height
            cv.rectangle(
                frame_rgb, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), 255, 2
            )

        # Display the mask and trackbars in the same window
        mask = resize(
            mask, 1 / scale_factor
        )  # Resize the mask back to the original size (assuming scale factor is scale_factor)
        mask_placeholder.image(mask)
        # display the base img with contours
        frame_placeholder.image(frame_rgb)
        # Wait for ESC key (27) to exit the loop
        if cv.waitKey(1) == 27 or stop_btn:
            break
    cap.release()
    cv.destroyAllWindows()


# Detect object color on images
def detect_object_color_image(
    frame,
    frame_placeholder,
    mask_placeholder,
    min_blue,
    min_green,
    min_red,
    max_blue,
    max_green,
    max_red,
):
    window_width = 800
    window_height = 600
    if frame is None:
        print("Cannot read image")
        exit()
    img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # Create masks for the specified HSV ranges
    mask = generate_range_color_mask(
        h, w, img, min_blue, max_blue, min_green, max_green, min_red, max_red
    )  # generate mask based on range colors
    result_contours = find_contours_bfs(mask)

    # Assuming largest means the contour with the most pixels
    if result_contours:
        largest_contour = max(result_contours, key=len)
        # Calculate the bounding box coordinates of the largest contour
        x_min = max(
            min(coord[1] for coord in largest_contour) - 3, 0
        )  # Ensure x_min is not negative
        y_min = max(
            min(coord[0] for coord in largest_contour) - 3, 0
        )  # Ensure y_min is not negative
        x_max = min(
            max(coord[1] for coord in largest_contour) + 3, img.shape[1] - 1
        )  # Ensure x_max is within image width
        y_max = min(
            max(coord[0] for coord in largest_contour) + 3, img.shape[0] - 1
        )  # Ensure y_max is within image height
        cv.rectangle(frame_rgb, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), 255, 2)

        # Display the mask and trackbars in the same window
        mask_placeholder.image(mask)
        # display the base img with contours
        frame_placeholder.image(frame_rgb)


# fonction pour générer l'image sans l'objet détecté
"""
@param image: l'image à modifier
@param background_image: l'image de fond pour remplacer l'objet
@param background_mask: le masque de l'objet détecté
return: image fusionnée
"""


def blend_images(image, background_image, background_mask):
    blended_image = np.zeros_like(
        image
    )  # Create the result image with same size as the original image
    h = image.shape[0]
    w = image.shape[1]

    y = 0
    while y < h:
        x = 0
        while x < w:
            if background_mask[y, x] == 0:  # If the object is not masked
                blended_image[y, x] = image[y, x]  # keep original pixel
            else:
                blended_image[y, x] = background_image[
                    y, x
                ]  # set result pixel to the background of the image
            x += 1
        y += 1

    return blended_image


def invisibility_cloak(
    cap,
    frame_placeholder,
    mask_placeholder,
    stop_btn,
    min_blue,
    min_green,
    min_red,
    max_blue,
    max_green,
    max_red,
):
    if not cap.isOpened():
        st.warning("Could not open video source", icon="")
        return
    for i in range(30):  # read the first 30 frames waiting for camera establishment
        ret, background = cap.read()  # keep the last frame as background

    scale_factor = 0.5
    background = resize_2(
        background, scale_factor
    )  # resize background for optimization

    while True and not stop_btn:
        _, frame = cap.read()  # get current frame
        frame_resized = resize_2(frame, scale_factor)  # resize frame for optimization
        h, w = frame.shape[:2]
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # convert frame to hsv
        hsv = resize_2(hsv, scale_factor)  # resize hsv frame
        h, w = hsv.shape[:2]

        # Generate the object to be invisible using object color detection object
        object_mask = generate_range_color_mask(
            h, w, hsv, min_blue, max_blue, min_green, max_green, min_red, max_red
        )

        # Replace the object detected with the background to be invisible
        blended_image = blend_images(frame_resized, background, object_mask)
        # Resize the result image to the original size
        blended_image = resize_2(blended_image, 1 / scale_factor)
        blended_image_rgb = cv.cvtColor(blended_image, cv.COLOR_BGR2RGB)
        # Display mask
        mask_placeholder.image(object_mask)
        # Display the invisibility cloak result
        frame_placeholder.image(blended_image_rgb)
        if cv.waitKey(1) == 27 or stop_btn:
            break

    cap.release()
    cv.destroyAllWindows()


def green_screen(
    cap,
    frame_placeholder,
    mask_placeholder,
    background_image,
    stop_btn,
    min_blue,
    min_green,
    min_red,
    max_blue,
    max_green,
    max_red,
):
    if not cap.isOpened():
        st.warning("Could not open video source", icon="")
        return
    _, frm = cap.read()  # get the current frame
    frm = resize_2(frm, 0.5)  # resize frame for optimization
    h, w, _ = frm.shape
    # get the background image to replace the green screen
    background_image = resize_2(
        background_image, 1, h, w
    )  # resize background image for optimization
    scale_factor = 0.5
    while True:
        ret, frame = cap.read()
        frame_resized = resize_2(frame, scale_factor)
        h, w = frame.shape[:2]
        hsv = cv.cvtColor(frame_resized, cv.COLOR_BGR2HSV)  # convert to HSV
        h, w = hsv.shape[:2]
        # Get the mask of green screen usong object color detection
        background_mask = generate_range_color_mask(
            h, w, hsv, min_blue, max_blue, min_green, max_green, min_red, max_red
        )

        # blend the original image and the background image
        blended_image = blend_images(frame_resized, background_image, background_mask)
        # resize image to the original size
        blended_image = resize_2(blended_image, 1 / scale_factor)
        # convert image to RGB to display it
        blended_image_rgb = cv.cvtColor(blended_image, cv.COLOR_BGR2RGB)
        # Display mask
        mask_placeholder.image(background_mask)
        # Display the result of green screen function
        frame_placeholder.image(blended_image_rgb)

        if cv.waitKey(1) == 27 or stop_btn:
            break

    cap.release()
    cv.destroyAllWindows()
