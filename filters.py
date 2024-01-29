import numpy as np
import cv2 as cv
from collections import deque
import streamlit as st

# **********************************************************************Filters Functions***********************************************
############################################Filtres de Lissages#######################################################

#######################Filtre mediane################


# Merging sort
def sort_array(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]  # Split the array into two halves
        right_half = arr[mid:]
        sort_array(left_half)  # Recursively sort the two halves
        sort_array(right_half)
        i = j = k = 0  # Initialize pointers for left, right, and merged halves

        while i < len(left_half) and j < len(
            right_half
        ):  # If did not reach the end of one half
            if left_half[i] < right_half[j]:  # Compare elements
                arr[k] = left_half[i]  # Store the smaller element
                i += 1
            else:
                arr[k] = right_half[j]  # Store the smaller element
                j += 1
            k += 1

        while i < len(left_half):  # If did not reach the end of left half
            arr[k] = left_half[i]  # Store the remaining elements
            i += 1
            k += 1

        while j < len(right_half):  # If did not reach the end of right half
            arr[k] = right_half[j]  # Store the remaining elements
            j += 1
            k += 1


# Uncolored median filter
def filtreMediane(img, voisinage):
    h, w = img.shape
    img_filtre = np.zeros((h, w), img.dtype)
    # loop without using the range function
    y = 0
    while y < h:
        x = 0
        while x < w:
            if (
                y < voisinage / 2
                or y > (h - voisinage / 2)
                or x < voisinage / 2
                or x > (w - voisinage / 2)
            ):
                img_filtre[y, x] = img[y, x]
            else:
                imgvois = img[
                    (y - voisinage // 2) : (y + voisinage // 2 + 1),
                    (x - voisinage // 2) : (x + voisinage // 2 + 1),
                ]
                moy = 0

                yv = 0
                yy, xx = imgvois.shape
                med = np.zeros((voisinage * voisinage), img.dtype)
                while yv < yy:
                    xv = 0
                    while xv < xx:
                        med[yv * xx + xv] = imgvois[yv, xv]
                        xv += 1
                    # med = sort_array(med)
                    med.sort()
                    img_filtre[y, x] = med[(voisinage * voisinage - 1) // 2]
                    yv += 1
            x += 1
        y += 1
    return img_filtre


# def filtreMediane(img,voisinage):
#     h,w = img.shape
#     img_filtre = np.zeros((h,w),img.dtype) # create a new image with the same size of the original image
#     #loop without using the range function
#     y = 0
#     while y <h:
#         x = 0
#         while x<w:
#             if y<voisinage/2 or y> (h-voisinage/2) or x<voisinage/2 or x>(w-voisinage/2): # if the pixel is a border pixel
#                 img_filtre[y,x] = img[y,x] # keep the original pixel
#             else: # if the pixel is not a border pixel
#                 imgvois = img[
#                     (y - voisinage // 2) : (y + voisinage // 2 + 1), # get the neighborhood of the pixel horizontally
#                     (x - voisinage // 2) : (x + voisinage // 2 + 1), # get the neighborhood of the pixel vertically
#                 ]

#                 yv = 0
#                 yy,xx = imgvois.shape
#                 med = np.zeros((voisinage*voisinage),img.dtype) # create a new array with the size of the neighborhood
#                 while yv <yy:
#                     xv = 0
#                     while xv<xx:
#                         med[yv*xx+xv] = imgvois[yv,xv] # fill the medianarray
#                         xv+=1
#                     sort_array(med) # sort the median array
#                     #med.sort()
#                     img_filtre[y,x] = med[(voisinage*voisinage-1)//2] # fill the filtered image with the median value
#                     yv+=1
#             x+=1
#         y+=1
#     return img_filtre # return the filtered image


# Colored median filter
def filtreMedianeColored(img, vois):
    # seperate every channel of the image
    b = img[:, :, 0]  # get the blue channel
    g = img[:, :, 1]  # get the green channel
    r = img[:, :, 2]  # get the red channel
    # apply the median filter to every channel
    b = filtreMediane(b, vois)  # apply the median filter to the blue channel
    g = filtreMediane(g, vois)  # apply the median filter to the green channel
    r = filtreMediane(r, vois)  # apply the median filter to the red channel

    # merge the filtered channels to get the final image
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


# Final median filter
def filtre_mediane(img, vois):
    if img.ndim == 3:  # if the image is colored
        return filtreMedianeColored(img, vois)
    else:  # if the image is grayscale
        return filtreMediane(img, vois)


########################################Filtre moyenne################


def filtre_moyen(img, voisinage):
    h, w = img.shape
    img_filtre = np.zeros(
        (h, w), img.dtype
    )  # create a new image with the same size of the original image
    # loop without using the range function
    y = 0
    while y < h:
        x = 0
        while x < w:
            if (
                y < voisinage / 2
                or y > (h - voisinage / 2)
                or x < voisinage / 2
                or x > (w - voisinage / 2)
            ):  # if the pixel is a border pixel
                img_filtre[y, x] = img[y, x]  # keep the original pixel
            else:
                imgvois = img[
                    (y - voisinage // 2) : (
                        y + voisinage // 2 + 1
                    ),  # get the neighborhood of the pixel horizontally
                    (x - voisinage // 2) : (
                        x + voisinage // 2 + 1
                    ),  # get the neighborhood of the pixel vertically
                ]
                moy = 0

                yv = 0
                yy, xx = imgvois.shape
                while yv < yy:
                    xv = 0
                    while xv < xx:
                        moy += imgvois[yv, xv]  # sum the values of the neighborhood
                        xv += 1
                    yv += 1
                img_filtre[y, x] = moy / (
                    voisinage * voisinage
                )  # fill the filtered image with the mean value
            x += 1
        y += 1
    return img_filtre


# Colored mean filter
def filtre_moyen_color(img, vois):
    # seperate every channel of the image
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    # apply the mean filter to every channel
    b = filtre_moyen(b, vois)
    g = filtre_moyen(g, vois)
    r = filtre_moyen(r, vois)
    # merge the filtered channels to get the final image
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


# Final mean filter
def filtreMoyen(img, vois):
    if img.ndim == 3:  # if the image is colored
        return filtre_moyen_color(img, vois)
    else:  # if the image is grayscale
        return filtre_moyen(img, vois)


############################################Filtres de convolution#######################################################


def filtre_gauss(img, sigma, taille_kernel):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert the image to grayscale
    t = taille_kernel
    kernel = np.zeros((t, t), np.float32)  # create the kernel
    sigma2 = sigma**2  # get the sigma squared
    const = 1 / (2 * np.pi * sigma2)  # get the constant
    y = 0
    x = 0
    while y < t:
        x = 0
        while x < t:
            kernel[y, x] = round(
                (const * np.exp(-((x - t / 2) ** 2 + (y - t / 2) ** 2) / (2 * sigma2))),
                4,
            )  # fill the kernel using the gaussian formula
            x += 1
        y += 1

    y = 0
    t = kernel.shape[0] // 2
    n = kernel.shape[0]
    h, w = img.shape
    imgRes = np.zeros(
        img.shape, img.dtype
    )  # create a new image with the same size of the original image

    while y < h:
        x = 0
        while x < w:
            if (
                y < t or y > h - t or x < t or x > w - t
            ):  # if the pixel is a border pixel
                imgRes[y, x] = img[y, x]  # keep the original pixel

            else:
                ky = 0

                while ky < n:
                    kx = 0
                    while kx < n:
                        imgRes[y, x] += (
                            img[y - ky, x - kx] * kernel[ky, kx]
                        )  # apply the convolution
                        kx += 1
                    ky += 1

            x += 1
        y += 1

    return imgRes


def filtre_laplacien(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert the image to grayscale
    im = img.copy()  # copy image in a new variable to work with
    h, w = im.shape
    y = 0
    while y < h:
        x = 0
        while x < w:
            # mask the image
            if im[y, x] > 125:  # if pixel is more likely to be edge
                im[y, x] = 255  # set the pixel to white
            else:
                im[y, x] = 0  # set the pixel to black
            x += 1
        y += 1

    kernel = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32
    )  # define laplacien kernel

    t = kernel.shape[0] // 2
    n = kernel.shape[0]
    h, w = im.shape
    imgRes = np.zeros(im.shape, im.dtype)

    y = 0
    while y < h:
        x = 0
        while x < w:
            if (
                y < t or y > h - t or x < t or x > w - t
            ):  # if the pixel is a border pixel
                imgRes[y, x] = im[y, x]  # keep the pixel of the mask

            else:
                ky = 0

                while ky < n:
                    kx = 0
                    while kx < n:
                        imgRes[y, x] += (
                            im[y - ky, x - kx] * kernel[ky, kx]
                        )  # apply the laplacien convolution
                        kx += 1
                    ky += 1

            x += 1
        y += 1

    return imgRes


#################################################Filtres morphologiques######################################################


# Add padding to the image
def add_pad(img, pad_height, pad_width):
    padded_image_height = img.shape[0] + 2 * pad_height  # add padding to the height
    padded_image_width = img.shape[1] + 2 * pad_width  # add padding to the width
    padded_image = np.zeros((padded_image_height, padded_image_width), dtype=np.uint8)
    for i in range(pad_height, padded_image_height - pad_height):
        for j in range(pad_width, padded_image_width - pad_width):
            padded_image[i][j] = img[i - pad_height][j - pad_width]
    return padded_image


# Dilate filter
def dilate_filter(image, kernel, threshold):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # convert image to grayscale
    binary_image = np.zeros(
        img.shape, dtype=np.uint8
    )  # create a new image with the same size of the original image
    # convert to binary
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # fill binary image
            if img[i][j] >= threshold:  # if the pixel is higher than the threshold
                binary_image[i][j] = 1  # set binary image to 1
            else:
                binary_image[i][j] = 0  # set binary image to 0
    height, width = binary_image.shape
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    # Pad the image with zeros to handle borders
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = add_pad(binary_image, pad_height, pad_width)

    # Initialize the result array
    result = np.zeros((height, width), dtype=np.uint8)

    # Apply dilate
    i = kernel_height // 2
    while i < height + kernel_height // 2:
        j = kernel_width // 2
        while j < width + kernel_width // 2:
            dilate = False
            for m in range(kernel_height):
                for n in range(kernel_width):
                    if (
                        kernel[m][n] == 1
                        and padded_image[i - kernel_height // 2 + m][
                            j - kernel_width // 2 + n
                        ]
                        == 1
                    ):
                        dilate = True  # If any overlapping condition meets, set dilate to True
                        break
                if dilate:  # If dilate is True, no need to continue inner loop
                    break
            if dilate:
                result[i - kernel_height // 2][j - kernel_width // 2] = 1
            else:
                result[i - kernel_height // 2][j - kernel_width // 2] = 0
            j += 1
        i += 1

    return result * 255


def erosion_filter(image, kernel, threshold):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # convert image to grayscale
    binary_image = np.zeros(img.shape, dtype=np.uint8)
    # convert to binary
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # fill binary image
            if img[i][j] >= threshold:  # if pixel higher than threshold
                binary_image[i][j] = 1  # set binary pixel to 1
            else:
                binary_image[i][j] = 0  # set binary pixel to 0
    height, width = binary_image.shape
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    # Pad the image with zeros to handle borders
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = add_pad(binary_image, pad_height, pad_width)

    # Initialize the result array
    result = np.zeros((height, width), dtype=np.uint8)

    # Apply erosion
    i = kernel_height // 2
    while i < height + kernel_height // 2:
        j = kernel_width // 2
        while j < width + kernel_width // 2:
            erosion = True
            for m in range(kernel_height):
                for n in range(kernel_width):
                    if (
                        kernel[m][n] == 1
                        and padded_image[i - kernel_height // 2 + m][
                            j - kernel_width // 2 + n
                        ]
                        == 0
                    ):
                        erosion = False
                        break
                if not erosion:
                    break

            if erosion:
                result[i - kernel_height // 2][j - kernel_width // 2] = 1
            else:
                result[i - kernel_height // 2][j - kernel_width // 2] = 0
            j += 1
        i += 1
    return result * 255


############################################# filtres seuillage #########################################################
def seuilage(img, seuil, max_value, type):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert image to grayscale
    dst = img.copy()
    # switch case the type : 0,1 ,2 3
    match type:
        case 0:
            # seuillage binaire
            x = 0
            while x < img.shape[0]:
                y = 0
                while y < img.shape[1]:
                    if img[x, y] > seuil:  # if pixel higher than threshold
                        dst[x, y] = max_value  # set pixel to max_value
                    else:
                        dst[x, y] = 0  # sel pixel to 0
                    y += 1
                x += 1
        case 1:
            # seuillage binaire inverse
            x = 0
            while x < img.shape[0]:
                y = 0
                while y < img.shape[1]:
                    if img[x, y] > seuil:
                        dst[x, y] = 0
                    else:
                        dst[x, y] = max_value
                    y += 1
                x += 1
        case 2:
            # seuillage troncature
            x = 0
            while x < img.shape[0]:
                y = 0
                while y < img.shape[1]:
                    if img[x, y] > seuil:  # if pixel higher than threshold
                        dst[x, y] = seuil  # set pixel to threshold
                    y += 1
                x += 1
        case 3:
            # seuillage a zero
            x = 0
            while x < img.shape[0]:
                y = 0
                while y < img.shape[1]:
                    if img[x, y] <= seuil:  # if pixel less than threshold
                        dst[x, y] = 0  # set pixel to 0
                    y += 1
                x += 1
        case _:
            dst = img
    return dst


###############################Additional filters###########################################


############################################# filtre sobel #########################################################
def filtre_sobel(img):
    # sobel filter
    # sobel kernels
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert image to grayscale
    kernelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # define kernel vertically
    kernely = np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    )  # define kernel horizontally
    h, w = img.shape
    y = 0
    dst = img.copy()
    while y < h - 1:
        x = 0
        while x < w - 1:
            # get the window array to apply the kernels
            window = img[y : y + 3, x : x + 3]
            hw, ww = window.shape
            yw = 0
            Gx = 0
            Gy = 0
            while yw < hw:
                xw = 0
                while xw < ww:
                    # multipty the window array with the kernels and sum the result in GX and GY
                    Gx += window[yw, xw] * kernelx[yw, xw]
                    Gy += window[yw, xw] * kernely[yw, xw]
                    xw += 1
                yw += 1
                # sum Gx and Gy and apply the sqrt
            res = np.sqrt(Gx * Gx + Gy * Gy)
            dst[y, x] = max(0, min(res, 255))
            x += 1
        y += 1

    return dst


#############################################  filtre emboss #########################################################
def filtre_emboss(img):
    # emboss filter
    # emboss kernels
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert image to grayscale
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])  # define emboss kernel
    h, w = img.shape
    y = 0
    dst = np.zeros((h, w), img.dtype)
    while y < h - 1:
        x = 0
        while x < w - 1:
            # get the window array to apply the kernels
            # check if the pixel is in the border
            if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                dst[y, x] = img[y, x]
            else:
                # put the pixel inthe middle of the window
                window = img[y - 1 : y + 2, x - 1 : x + 2]
                hw, ww = window.shape
                yw = 0
                sumw = 0
                while yw < hw:
                    xw = 0
                    while xw < ww:
                        # multipty the window array with the kernels and sum the result in GX and GY
                        sumw += window[yw][xw] * kernel[yw][xw]
                        xw += 1
                    yw += 1

                dst[y][x] = max(0, min(255, sumw))
            x += 1
        y += 1
    return dst
