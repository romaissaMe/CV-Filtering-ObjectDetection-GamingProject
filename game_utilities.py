import numpy as np
import cv2 as cv
from collections import deque
from utilities import *
from filters import filtre_moyen_color


class Ghost:
    def __init__(self, position, speed):
        self.position = position
        self.speed = speed

    def update_position(self):
        # Update the ghost's position based on its speed
        self.position = (self.position[0] + self.speed, self.position[1])


class Cherry:
    def __init__(self, position, speed):
        self.position = position
        self.speed = speed

    def update_position(self):
        # Update the cherry's position based on its speed
        self.position = (self.position[0] + self.speed, self.position[1])


class Throwables:
    def __init__(
        self,
        frame,
        difficulty,
        cherry_img,
        ghost_img,
        cherry_width,
        cherry_height,
        ghost_width,
        ghost_height,
    ):
        self.ghosts = [
            Ghost((0, np.random.randint(0, frame.shape[1])), difficulty // 2)
        ]  # intialize ghost
        self.speed = difficulty // 2
        self.difficulty = int((1 / difficulty) * 800)
        print("diff", self.difficulty)
        self.cherries = []  # initialize cherries
        self.cherry_width = cherry_width
        self.cherry_height = cherry_height
        self.ghost_width = ghost_width
        self.ghost_height = ghost_height
        self.cherry_img = cherry_img
        self.ghost_img = ghost_img

    def add_cherry(self, frame, score):
        # add a cherry randomly depending on the score
        if np.random.randint(0, 800) < score and len(self.cherries) < 1:
            self.cherries.append(
                Cherry((0, np.random.randint(0, frame.shape[1])), self.speed)
            )

    def update_positions(self, frame):
        # update the position for the cherries and the ghosts
        for cherry in self.cherries:
            prev_pos = cherry.position
            cherry.update_position()
            frame[
                prev_pos[0] : prev_pos[0] + self.cherry_height,
                prev_pos[1] : prev_pos[1] + self.cherry_width,
            ] = 0
            # if the cherry is out of the frame, remove it
            if (
                cherry.position[0] > frame.shape[0] - self.cherry_height
                or cherry.position[0] < 0
                or cherry.position[1] > frame.shape[1] - self.cherry_width
                or cherry.position[1] < 0
            ):
                self.cherries.remove(cherry)
            else:
                # remove the cherry from the previous position
                frame[
                    cherry.position[0] : cherry.position[0] + self.cherry_height,
                    cherry.position[1] : cherry.position[1] + self.cherry_width,
                ] = self.cherry_img
        for ghost in self.ghosts:
            prev_pos = ghost.position
            ghost.update_position()
            frame[
                prev_pos[0] : prev_pos[0] + self.ghost_height,
                prev_pos[1] : prev_pos[1] + self.ghost_width,
            ] = 0
            # if the ghost is out of the frame, remove it
            if (
                ghost.position[0] > frame.shape[0] - self.ghost_height
                or ghost.position[0] < 0
                or ghost.position[1] > frame.shape[1] - self.ghost_width
                or ghost.position[1] < 0
            ):
                self.ghosts.remove(ghost)
            else:
                # remove the ghost from the previous position
                frame[
                    ghost.position[0] : ghost.position[0] + self.ghost_height,
                    ghost.position[1] : ghost.position[1] + self.ghost_width,
                ] = self.ghost_img

    def update_ghosts(self, score, frame):
        # update the number of ghosts based on the score
        if np.random.randint(0, self.difficulty) < score:
            self.ghosts.append(
                Ghost((0, np.random.randint(0, frame.shape[1])), self.speed)
            )

    def check_cherry_collision(
        self, frame, pacman_position, pacman_height, pacman_width
    ):
        # Check if Pacman's position overlaps with any cherry's position
        for cherry in self.cherries:
            dist = abs(pacman_position[0] - cherry.position[0]) + abs(
                pacman_position[1] - cherry.position[1]
            )
            if dist < pacman_height:
                print(dist)
                print("pacamn pos", pacman_position, "cherry pos", cherry.position)
                # clear the cherry from the frame
                frame[
                    cherry.position[0] : cherry.position[0] + self.cherry_height,
                    cherry.position[1] : cherry.position[1] + self.cherry_width,
                ] = 0
                self.cherries.remove(cherry)
                self.speed += 1
                return True
        return False

    def check_collision(self, pacman_position, pacman_height, pacman_width):
        # Check if Pacman's position overlaps with any ghost's position
        for ghost in self.ghosts:
            dist = abs(pacman_position[0] - ghost.position[0]) + abs(
                pacman_position[1] - ghost.position[1]
            )
            if dist < pacman_height:
                print(dist)
                print("pacamn pos", pacman_position, "ghost pos", ghost.position)
                return True
        return False


def draw_img(frame, img, centroid, prev_pos, min_h):
    # clear the previous position of pacman
    img_height, img_width, _ = img.shape
    h, w, _ = frame.shape
    new_pose = centroid
    # Calculate the lowest position the image of Pacman can reach
    min_height = int(frame.shape[0] * min_h)
    if prev_pos:
        frame[
            prev_pos[0] : prev_pos[0] + img_height,
            prev_pos[1] : prev_pos[1] + img_width,
        ] = 0
    # Check if the image of Pacman fits within the frame
    if (
        centroid[0] + img_height <= frame.shape[0]
        and centroid[1] + img_width <= frame.shape[1]
        and centroid[0] >= 0
        and centroid[1] >= 0
    ):
        # Check if the centroid of the largest contour is within the last 30% of the frame height
        if centroid[0] >= min_height:
            # Display the image of Pacman at the centroid of the largest contour
            frame[
                centroid[0] : centroid[0] + img_height,
                centroid[1] : centroid[1] + img_width,
            ] = img
        else:
            #   Display the image of Pacman at the lowest position it can reach
            frame[
                min_height : min_height + img_height,
                centroid[1] : centroid[1] + img_width,
            ] = img

            new_pose = (min_height, centroid[1])
        return frame, new_pose
    else:
        # Display the image of Pacman at the lowest position it can reach
        # check if the x coordinate is out of the frame and correct it
        if centroid[0] < 0:
            centroid[0] = 0
        if centroid[1] < 0:
            centroid[1] = 0
        if centroid[1] + img_width >= frame.shape[1]:
            centroid[1] = frame.shape[1] - img_width
        # check if the y coordinate is out of the frame and correct it
        if centroid[0] + img_height >= frame.shape[0]:
            centroid[0] = frame.shape[0] - img_height
        frame[
            centroid[0] : centroid[0] + img_height,
            centroid[1] : centroid[1] + img_width,
        ] = img
        new_pose = (centroid[0], centroid[1])
    return frame, new_pose


def update_position_with_keys(key, game_array, pacman_img, pacman_position, step):
    if key in [97, 65]:  # Left arrow
        key_position = [pacman_position[0], pacman_position[1] - step]
        game_array, pacman_position = draw_img(
            game_array, pacman_img, key_position, pacman_position, 0.65
        )
    elif key in [119, 87]:  # Up arrow
        key_position = [pacman_position[0] - step, pacman_position[1]]
        game_array, pacman_position = draw_img(
            game_array, pacman_img, key_position, pacman_position, 0.65
        )
    elif key in [100, 68]:  # Right arrow
        key_position = [pacman_position[0], pacman_position[1] + step]
        game_array, pacman_position = draw_img(
            game_array, pacman_img, key_position, pacman_position, 0.65
        )
    elif key in [115, 83]:  # Down arrow
        key_position = [pacman_position[0] + step, pacman_position[1]]
        game_array, pacman_position = draw_img(
            game_array, pacman_img, key_position, pacman_position, 0.65
        )
    return game_array, pacman_position


def resize(image, scale_factor):
    height, width, channels = image.shape

    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            y = int(i / scale_factor)
            x = int(j / scale_factor)
            new_image[i][j] = image[y][x]

    return new_image


def play_game(
    scale_factor,
    height_limit,
    player_movement,
    vois,
    difficulty,
    pacman_img,
    pacman_height,
    pacman_width,
    cherry_img,
    ghost_img,
    cherry_width,
    cherry_height,
    ghost_width,
    ghost_height,
):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    cv.namedWindow("Track Bars", cv.WINDOW_NORMAL)
    window_width = 800
    window_height = 600
    cv.resizeWindow("Track Bars", window_width, window_height)
    # create trackbars for BGR ranges
    colors = ["blue", "green", "red"]
    for color in colors:
        cv.createTrackbar(f"min_{color}", "Track Bars", 80, 255, nothing)

    for color in colors:
        cv.createTrackbar(f"max_{color}", "Track Bars", 110, 255, nothing)
    start_game = False
    frame_count = 0
    score = 1
    h, w, _ = cap.read()[1].shape
    score_layer = np.ones((50, 150, 3), np.uint8)
    # initiate the ghosts
    ghosts = Throwables(
        cap.read()[1],
        difficulty,
        cherry_img,
        ghost_img,
        cherry_width,
        cherry_height,
        ghost_width,
        ghost_height,
    )
    height, width, _ = cap.read()[1].shape
    game_array = np.zeros((height, width, 3), np.uint8)
    # draw a blue line on the game array
    game_array[int(height * height_limit), :, :] = (255, 0, 0)
    pacman_position = (0, 0)
    while True:
        ret, frame = cap.read()
        resized_frame = resize(frame, scale_factor)
        hsv = cv.cvtColor(resized_frame, cv.COLOR_BGR2HSV)

        if not ret:
            break
        if cv.waitKey(1) & 0xFF == ord(" "):
            start_game = True
            print("Game Started")
        h, w, _ = hsv.shape
        min_blue = cv.getTrackbarPos("min_blue", "Track Bars")
        min_green = cv.getTrackbarPos("min_green", "Track Bars")
        min_red = cv.getTrackbarPos("min_red", "Track Bars")

        max_blue = cv.getTrackbarPos("max_blue", "Track Bars")
        max_green = cv.getTrackbarPos("max_green", "Track Bars")
        max_red = cv.getTrackbarPos("max_red", "Track Bars")

        # Create masks for the specified HSV ranges
        mask = generate_range_color_mask(
            h, w, hsv, min_blue, max_blue, min_green, max_green, min_red, max_red
        )
        result_contours = find_contours_bfs(mask)
        # Assuming largest means the contour with the most pixels
        centroid = None
        if result_contours:
            largest_contour = max(result_contours, key=len)
            # the largest contour is calculated using the resized frame, so we need to rescale the largest contour to the original size
            largest_contour = [
                (int(y / scale_factor), int(x / scale_factor))
                for y, x in largest_contour
            ]

            # Calculate the centroid of the largest contour
            centroid_x = int(
                sum(coord[1] for coord in largest_contour) / len(largest_contour)
            )
            centroid_y = int(
                sum(coord[0] for coord in largest_contour) / len(largest_contour)
            )
            centroid = [centroid_y, centroid_x]
            # draw the pacman image
            res, pacman_position = draw_img(
                game_array, pacman_img, centroid, pacman_position, height_limit
            )
            if res is not None:
                game_array = res

        game_array[int(height * height_limit), :, :] = (255, 0, 0)
        key = cv.waitKey(1)
        game_array, pacman_position = update_position_with_keys(
            key, game_array, pacman_img, pacman_position, player_movement
        )
        if start_game:
            # Update the score every 400 frames
            if frame_count == 400:
                score += 1
                print(score)
                frame_count = 0
                ghosts.speed += 1
            frame_count += 1
            if ghosts.check_cherry_collision(
                game_array, pacman_position, pacman_height, pacman_width
            ):
                score += 4
            # Update the positions of the ghosts
            ghosts.update_positions(game_array)
            # add cherry
            ghosts.add_cherry(game_array, score)
            # Display the score
            score_layer = np.ones((30, 150, 3), np.uint8)
            cv.putText(
                score_layer,
                f"Score: {score}",
                (20, 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )

            ghosts.update_ghosts(score, game_array)
            # Check if the pacman collided with any of the ghosts
            if ghosts.check_collision(pacman_position, pacman_height, pacman_width):
                # End the game
                print("Game Over")
                break
        game_array[0 : score_layer.shape[0], 0 : score_layer.shape[1]] = score_layer
        # Display the frame
        cv.imshow("frame", frame)
        cv.imshow("game", game_array)
        cv.imshow("Track Bars", mask)
        # Break the loop if the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv.destroyAllWindows()
