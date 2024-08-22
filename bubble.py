import cv2
import mediapipe as mp
import random
import time
import pygame  # Import pygame for sound handling

# Initialize pygame mixer for sound and music
pygame.mixer.init()

# Load the pop sound effect
pop_sound = pygame.mixer.Sound('pop.mp3')

# Load and play background music
pygame.mixer.music.load('beach-music.mp3')
pygame.mixer.music.play(-1)  # Play the music indefinitely

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Set the resolution of the capture window to 1920x1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Load the bubble image
bubble_image = cv2.imread('gelembung.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Load the start and quit button images
start_button_image = cv2.imread('start.png', cv2.IMREAD_UNCHANGED)
quit_button_image = cv2.imread('quit.png', cv2.IMREAD_UNCHANGED)

# Button sizes
button_width = 400
button_height = 150

# Button positions for side-by-side layout
start_button_pos = (560, 450)  # Adjusted position for the start button
quit_button_pos = (960, 450)   # Adjusted position for the quit button, next to the start button

# Game state
game_started = False

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay `img_overlay` on top of `img` at the position `pos`
    and blend using `alpha_mask`."""
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, None]

    img_crop[:] = alpha * img_overlay_crop + (1 - alpha) * img_crop

# Bubble configuration
class Bubble:
    def __init__(self, x, y, radius, speed):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        self.alive = True

    def reset(self):
        """Reset the bubble to start from the top again with a random horizontal position."""
        self.y = -self.radius
        self.x = random.randint(50, 1870)
        self.alive = True

    def move(self):
        self.y += self.speed
        if self.y - self.radius > h:  # If the bubble moves off the screen
            self.reset()  # Reset the bubble's position

    def draw(self, frame):
        if self.alive:
            # Resize the bubble image to match the radius
            resized_bubble = cv2.resize(bubble_image, (self.radius * 2, self.radius * 2))

            # Split the channels
            b, g, r, a = cv2.split(resized_bubble)

            # Create a mask from the alpha channel
            mask = cv2.merge((a, a, a))
            mask = mask / 255.0  # Normalize the mask to be between 0 and 1

            # Prepare the RGB image (without alpha)
            bubble_rgb = cv2.merge((b, g, r))

            # Calculate the position to place the image so that it's centered on the bubble's coordinates
            top_left_x = int(self.x - self.radius)
            top_left_y = int(self.y - self.radius)
            bottom_right_x = top_left_x + resized_bubble.shape[1]
            bottom_right_y = top_left_y + resized_bubble.shape[0]

            # Ensure the bubble is drawn within the frame boundaries
            if top_left_x >= 0 and top_left_y >= 0 and bottom_right_x <= frame.shape[1] and bottom_right_y <= frame.shape[0]:
                # Extract the region of interest from the frame where the bubble will be drawn
                roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                # Blend the bubble image with the frame using the mask
                roi = roi * (1 - mask) + bubble_rgb * mask

                # Place the blended image back into the frame
                frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = roi

    def check_collision(self, hand_x, hand_y):
        distance = ((self.x - hand_x) ** 2 + (self.y - hand_y) ** 2) ** 0.5
        if distance < self.radius:
            self.alive = False
            pop_sound.play()  # Play the pop sound when a bubble is popped
            return True
        return False

# Bubble setup and counter
initial_speed = 5
bubble_count = 10
bubbles = [Bubble(random.randint(50, 1870), 0, random.randint(20, 50), initial_speed) for _ in range(bubble_count)]
score = 0
speed_increase_interval = 30  # seconds
last_speed_increase_time = time.time()

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        if not ret:
            print("Failed to capture frame from camera.")
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
                hand_landmarks.append((int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                                       int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)))

        if not game_started:
            # Resize button images
            resized_start_button = cv2.resize(start_button_image, (button_width, button_height))
            resized_quit_button = cv2.resize(quit_button_image, (button_width, button_height))

            # Split the RGBA channels
            start_b, start_g, start_r, start_a = cv2.split(resized_start_button)
            quit_b, quit_g, quit_r, quit_a = cv2.split(resized_quit_button)

            # Merge RGB channels back together
            start_rgb = cv2.merge((start_b, start_g, start_r))
            quit_rgb = cv2.merge((quit_b, quit_g, quit_r))

            # Overlay the start and quit buttons side by side
            overlay_image_alpha(frame, start_rgb, start_button_pos, start_a / 255.0)
            overlay_image_alpha(frame, quit_rgb, quit_button_pos, quit_a / 255.0)

            # Check if the hand is pointing to the start or quit button
            for hand_pos in hand_landmarks:
                if start_button_pos[0] < hand_pos[0] < start_button_pos[0] + button_width and start_button_pos[1] < hand_pos[1] < start_button_pos[1] + button_height:
                    game_started = True  # Start the game
                if quit_button_pos[0] < hand_pos[0] < quit_button_pos[0] + button_width and quit_button_pos[1] < hand_pos[1] < quit_button_pos[1] + button_height:
                    cap.release()
                    cv2.destroyAllWindows()
                    quit()  # Exit the program
        else:
            # Check for collisions with bubbles
            for bubble in bubbles:
                if bubble.alive:
                    for hand_pos in hand_landmarks:
                        if bubble.check_collision(hand_pos[0], hand_pos[1]):
                            score += 1
                            bubble.reset()  # Reset the bubble after it is popped
                    bubble.move()
                    bubble.draw(frame)
                else:
                    bubble.reset()  # Reset the bubble if it's not alive (i.e., was popped)

            # Increase bubble speed every 30 seconds
            current_time = time.time()
            if current_time - last_speed_increase_time >= speed_increase_interval:
                for bubble in bubbles:
                    bubble.speed += 5  # Increase the speed of each bubble
                last_speed_increase_time = current_time  # Reset the timer

        cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Bubble Pop', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
