################################# Utility Functions ########################################
import cv2 as cv
from collections import deque
import numpy as np


# in this code we use generate_range_color_mask
# generate_mask compares pixel = exact color
def generate_range_color_mask(
    h, w, img_hsv, min_blue, max_blue, min_green, max_green, min_red, max_red
):
    hx = 0
    wx = 0
    mask = np.zeros((h, w), np.uint8)
    min_rgb_colors = np.array([[[min_blue, min_green, min_red]]], dtype=np.uint8)
    min_hsv_colors = cv.cvtColor(
        min_rgb_colors, cv.COLOR_BGR2HSV
    )  # convert min values to HSV
    h_min, s_min, v_min = min_hsv_colors[0][0]
    max_rgb_colors = np.array([[[max_blue, max_green, max_red]]], dtype=np.uint8)
    max_hsv_colors = cv.cvtColor(
        max_rgb_colors, cv.COLOR_BGR2HSV
    )  # convert max values to HSV
    h_max, s_max, v_max = max_hsv_colors[0][0]
    while hx < h:
        wx = 0
        while wx < w:
            # if pixel in range of min and max values
            if (
                h_min <= img_hsv[hx, wx][0] < h_max
                and s_min <= img_hsv[hx, wx][1] < s_max
                and v_min <= img_hsv[hx, wx][2] < v_max
            ):
                mask[hx, wx] = 255  # set mask to white
            else:
                mask[hx, wx] = 0  # set mask to black
            wx += 1
        hx += 1
    return mask


### finding contours of the detected object in an image to help draw a rectangle
def find_contours_bfs(image):
    height, width = len(image), len(image[0])
    # using a set is better than using an array
    visited = set()
    contours = []

    def bfs(x, y):
        # same for the deque
        queue = deque([(x, y)])
        contour = [(x, y)]
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (
                    nx >= 0
                    and ny >= 0
                    and nx < height
                    and ny < width
                    and image[nx][ny] == 255
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    contour.append((nx, ny))
        print("vision")
        return contour

    for i in range(height):
        for j in range(width):
            if image[i][j] == 255 and (i, j) not in visited:
                visited.add((i, j))
                contour = bfs(i, j)
                contours.append(contour)

    return contours


# the resize function, to resize the image and keep the aspect ratio
def resize(image, scale_factor):
    if image.ndim == 3:
        height, width, channels = image.shape
        # multipy the height and width by the scale factor
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        # create a new image with the new height and width
        new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)
        # loop over the new image and fill it with the pixels of the old image
        for i in range(new_height):
            for j in range(new_width):
                y = int(i / scale_factor)
                x = int(j / scale_factor)
                new_image[i][j] = image[y][x]
        return new_image
    elif image.ndim == 2:
        height, width = image.shape
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        new_image = np.zeros((new_height, new_width), dtype=image.dtype)
        for i in range(new_height):
            for j in range(new_width):
                y = int(i / scale_factor)
                x = int(j / scale_factor)
                new_image[i][j] = image[y][x]
        return new_image


# Fonction 2 pour redimensionner les images
"""
@param image: l'image à redimensionner
@param scale_factor: l'échelle de redimensionnement
@param new_height: la nouvelle hauteur
@param new_width: la nouvelle largeur
return: l'image redimensionné
"""


def resize_2(
    image, scale_factor, new_height=None, new_width=None
):  # scale_factor=1 if new_height and new_width != None
    height, width, channels = image.shape

    if new_height is None and new_width is None:
        # If values==None, calculate new height and new width using the scale factor
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
    elif new_height is None:
        # If height==None, calculate new height using scale factor
        new_height = int(new_width * (height / width) * scale_factor)
    elif new_width is None:
        # if width==None, calculate new width using scale factor
        new_width = int(new_height * (width / height) * scale_factor)
    # Else we use the specified height and width in parameters

    # Create new image with new height and new width
    new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)

    # Loop on the new image to fill it
    for i in range(new_height):
        for j in range(new_width):
            y = int(
                i / scale_factor
            )  # get the correspondent pixel from the original image
            x = int(j / scale_factor)
            new_image[i][j] = image[y][x]

    return new_image


###################################Creation des images pour le jeu###########################################
def create_packman():
    # Créer une image noire de taille 200x200 avec 3 canaux de couleur (pour RGB)
    height, width = 200, 200
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Dessiner Pac-Man avec des pixels jaunes
    # Coordonnées du centre de Pac-Man
    center_x, center_y = 100, 100

    # Rayon de Pac-Man
    radius = 80

    # Boucle pour dessiner Pac-Man
    y = 0
    while y < height:
        x = 0
        while x < width:
            # Calculer la distance entre le pixel actuel et le centre de Pac-Man
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Vérifier si le pixel est à l'intérieur du cercle représentant Pac-Man
            if distance <= radius:
                # Colorier le pixel en jaune pour représenter Pac-Man (RGB : 255, 255, 0)
                image[y, x] = [0, 255, 255]

            # Dessiner la bouche (triangle manquant)
            # On vérifie si le pixel est dans la partie manquante du cercle pour dessiner la bouche
            if distance <= radius and x > center_x and y <= center_y:
                # Calculer l'équation de la ligne pour dessiner un triangle représentant la bouche
                slope = -1 * (center_y - y) / (center_x - x)
                if slope >= -1 and slope <= 1:
                    # Si le pixel est dans la partie manquante, le laisser noir (bouche ouverte)
                    image[y, x] = [0, 0, 0]

            # Dessiner l'œil de Pac-Man
            eye_x, eye_y = 120, 80
            eye_radius = 8

            # Coordonnées du point noir pour l'œil
            eye_center_x, eye_center_y = 100, 60

            # Vérifier si le pixel est à l'intérieur du cercle représentant l'œil
            eye_distance = np.sqrt((x - eye_center_x) ** 2 + (y - eye_center_y) ** 2)
            if eye_distance <= eye_radius:
                # Colorier le pixel en noir pour représenter l'œil de Pac-Man (RGB : 0, 0, 0)
                image[y, x] = [0, 0, 0]

            x += 1
        y += 1

    # Afficher l'image de Pac-Man
    cv.imshow("Pac-Man Pixel Art", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Enregistrer l'image au format PNG
    cv.imwrite("pacman.png", image)


def create_ghost():
    # Créer une image noire de taille 200x200 avec 3 canaux de couleur (pour RGB)
    height, width = 200, 200
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Dessiner un fantôme avec des pixels cyan
    # Coordonnées du centre du demi-cercle représentant la tête du fantôme
    head_center_x, head_center_y = 100, 80

    # Rayon du demi-cercle pour la tête du fantôme
    head_radius = 40

    # Boucle pour dessiner le demi-cercle représentant la tête du fantôme
    for y in range(height):
        for x in range(width):
            # Calculer la distance entre le pixel actuel et le centre de la tête du fantôme
            distance = np.sqrt((x - head_center_x) ** 2 + (y - head_center_y) ** 2)

            # Vérifier si le pixel est à l'intérieur du cercle représentant la tête du fantôme
            if distance <= head_radius and y <= head_center_y:
                # Colorier le pixel en cyan pour représenter la tête du fantôme (RGB : 0, 255, 255)
                image[y, x] = [0, 255, 255]

    # Dessiner le rectangle pour le bas du corps du fantôme
    # Coordonnées du coin supérieur gauche du rectangle
    rect_top_left_x, rect_top_left_y = 60, 100
    # Largeur et hauteur du rectangle
    rect_width, rect_height = 40, 40

    # Boucle pour dessiner le rectangle représentant le bas du corps du fantôme
    for y in range(rect_top_left_y, rect_top_left_y + rect_height):
        for x in range(rect_top_left_x, rect_top_left_x + rect_width):
            # Colorier les pixels en cyan pour représenter le bas du corps du fantôme
            image[y, x] = [0, 255, 255]

    # Afficher l'image du fantôme
    cv.imshow("Ghost Pixel Art", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Enregistrer l'image au format PNG
    cv.imwrite("ghost.png", image)


def create_cherry():
    # Create a black image of size 200x200 with 3 color channels (for RGB)
    height, width = 200, 200
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw cherry with red circles and green rectangles
    # Define cherry parameters
    center_circle1 = (90, 90)
    center_circle2 = (110, 90)
    radius = 30
    thickness = 1  # Thickness of the lines to simulate the circles

    # Draw the first red circle for the cherry
    for y in range(height):
        for x in range(width):
            if (x - center_circle1[0]) ** 2 + (
                y - center_circle1[1]
            ) ** 2 <= radius**2:
                image[y, x] = [0, 0, 255]  # Set pixel color to red

    # Draw the second red circle close to the first one
    for y in range(height):
        for x in range(width):
            if (x - center_circle2[0]) ** 2 + (
                y - center_circle2[1]
            ) ** 2 <= radius**2:
                image[y, x] = [0, 0, 255]  # Set pixel color to red

    # Draw vertical rectangles above each circle
    rect1_top_left = (80, 70)
    rect1_bottom_right = (100, 90)
    rect2_top_left = (110, 70)
    rect2_bottom_right = (130, 90)

    for y in range(rect1_top_left[1], rect1_bottom_right[1]):
        for x in range(rect1_top_left[0], rect1_bottom_right[0]):
            image[y, x] = [0, 255, 0]  # Set pixel color to green

    for y in range(rect2_top_left[1], rect2_bottom_right[1]):
        for x in range(rect2_top_left[0], rect2_bottom_right[0]):
            image[y, x] = [0, 255, 0]  # Set pixel color to green

    # Draw horizontal rectangle between the vertical rectangles
    rect3_top_left = (100, 80)
    rect3_bottom_right = (110, 85)

    for y in range(rect3_top_left[1], rect3_bottom_right[1]):
        for x in range(rect3_top_left[0], rect3_bottom_right[0]):
            image[y, x] = [0, 255, 0]  # Set pixel color to green

    # Display the cherry image
    cv.imshow("Cherry Pixel Art", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save the image as a PNG file
    cv.imwrite("cherry_test.png", image)


# just used in trackbar function
def nothing(x):
    pass
