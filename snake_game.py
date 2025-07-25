import cv2
import mediapipe as mp
import pygame
import random
import sys
import numpy as np
import threading
import time
import os

# --- Pygame & Sound Initialization ---
pygame.init()
pygame.mixer.init()

# --- Game Window Settings (Fullscreen) ---
infoObject = pygame.display.Info()
WIDTH, HEIGHT = infoObject.current_w, infoObject.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Gesture Controlled Snake")
clock = pygame.time.Clock()

# **FIX**: Slightly larger grid size for better visuals
GRID_SIZE = 30
# **FIX**: Calculate a perfectly aligned game area to avoid visual glitches at the borders
GRID_WIDTH = (WIDTH // GRID_SIZE) - 2
GRID_HEIGHT = (HEIGHT // GRID_SIZE) - 2
game_area_width = GRID_WIDTH * GRID_SIZE
game_area_height = GRID_HEIGHT * GRID_SIZE
start_x = (WIDTH - game_area_width) // 2
start_y = (HEIGHT - game_area_height) // 2
GAME_AREA_RECT = pygame.Rect(start_x, start_y, game_area_width, game_area_height)

# --- Colors and Fonts ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
SNAKE_BODY_COLOR = (0, 200, 0)
SNAKE_HEAD_COLOR = (0, 220, 100)
INVINCIBLE_SNAKE_COLOR = (255, 200, 0)
GRID_COLOR = (40, 40, 40, 150)  # Added alpha for transparency
ZONE_COLOR = (0, 100, 255, 100)
ZONE_ACTIVE_COLOR = (0, 255, 100, 150)
FONT = pygame.font.SysFont('Consolas', 25)
ARROW_FONT = pygame.font.SysFont('Arial', 40, bold=True)
GAME_OVER_FONT = pygame.font.SysFont('Consolas', 60)
TITLE_FONT = pygame.font.SysFont('Consolas', 80, bold=True)
PAUSE_FONT = pygame.font.SysFont('Consolas', 100, bold=True)

# --- Asset and High Score Management ---
ASSETS_DIR = "assets"
HIGHSCORE_FILE = os.path.join(ASSETS_DIR, "highscore.txt")


def load_highscore():
    # Create assets directory if it doesn't exist
    if not os.path.isdir(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
    if os.path.exists(HIGHSCORE_FILE):
        with open(HIGHSCORE_FILE, 'r') as f:
            try:
                return int(f.read())
            except ValueError:
                return 0
    return 0


def save_highscore(score):
    # Ensure directory exists before saving
    if not os.path.isdir(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
    with open(HIGHSCORE_FILE, 'w') as f: f.write(str(score))


def load_sound(filename):
    try:
        return pygame.mixer.Sound(os.path.join(ASSETS_DIR, filename))
    except pygame.error:
        print(f"Warning: Sound file not found: {os.path.join(ASSETS_DIR, filename)}. Game will run without this sound.")
        return None


sound_crunch = load_sound("crunch.wav")
sound_powerup = load_sound("powerup.wav")
sound_game_over = load_sound("game_over.wav")


# --- Asset Creation ---
def create_apple_surface(size):
    apple_surface = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(apple_surface, (255, 0, 0), (size // 2, size // 2), size // 2)
    pygame.draw.rect(apple_surface, (139, 69, 19), (size // 2 - 2, 0, 4, size // 3))
    pygame.draw.ellipse(apple_surface, (50, 205, 50), (size // 2 - 10, 5, 12, 6))
    return apple_surface


# --- Thread-Safe Shared Data ---
latest_frame = None
game_state = 'intro'
active_zone = None
frame_lock = threading.Lock()
state_lock = threading.Lock()
running = True


# --- Hand Tracking Thread ---
def hand_tracking_worker():
    global latest_frame, game_state, active_zone, running
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): running = False; return

    zone_width, zone_height = WIDTH // 4, HEIGHT // 4
    zones = {
        'up': pygame.Rect(zone_width, 0, zone_width * 2, zone_height),
        'down': pygame.Rect(zone_width, HEIGHT - zone_height, zone_width * 2, zone_height),
        'left': pygame.Rect(0, zone_height, zone_width, zone_height * 2),
        'right': pygame.Rect(WIDTH - zone_width, zone_height, zone_width, zone_height * 2)
    }

    while running:
        success, frame = cap.read()
        if not success: time.sleep(0.01); continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with frame_lock:
            latest_frame = rgb_frame.copy()
        results = hands.process(rgb_frame)
        current_active_zone, new_direction, gesture_detected = None, None, 'none'

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # **FIX**: More robust gesture detection logic
            # Get landmarks for all relevant fingers
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

            # Check if each finger is extended (tip is above the pip joint)
            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y
            ring_up = ring_tip.y < ring_pip.y
            pinky_up = pinky_tip.y < pinky_pip.y

            # Determine gesture based on finger states
            if index_up and middle_up and ring_up and pinky_up:
                gesture_detected = 'pause'  # Open palm
            elif index_up and not middle_up and not ring_up and not pinky_up:
                gesture_detected = 'play'  # Pointing with only index finger

            finger_pos = (index_tip.x * WIDTH, index_tip.y * HEIGHT)
            if zones['up'].collidepoint(finger_pos):
                new_direction, current_active_zone = (0, -1), 'up'
            elif zones['down'].collidepoint(finger_pos):
                new_direction, current_active_zone = (0, 1), 'down'
            elif zones['left'].collidepoint(finger_pos):
                new_direction, current_active_zone = (-1, 0), 'left'
            elif zones['right'].collidepoint(finger_pos):
                new_direction, current_active_zone = (1, 0), 'right'

        with state_lock:
            active_zone = current_active_zone
            if game_state == 'intro' and gesture_detected == 'play':
                game_state = 'playing'
            elif game_state == 'playing' and gesture_detected == 'pause':
                game_state = 'paused'
            elif game_state == 'paused' and gesture_detected == 'play':
                game_state = 'playing'
            if game_state == 'playing' and new_direction: snake.change_direction(new_direction)
        time.sleep(1 / 30)
    cap.release()


# --- Game Classes ---
class Snake:
    def __init__(self):
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (0, 0)
        self.grow = False
        self.invincible = False
        self.invincibility_end_time = 0

    def change_direction(self, new_dir):
        if self.direction != (0, 0) and (new_dir[0] * -1, new_dir[1] * -1) == self.direction: return
        self.direction = new_dir

    def grow_snake(self):
        self.grow = True

    def move(self):
        if self.direction == (0, 0): return True
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        if self.invincible:
            new_head = (new_head[0] % GRID_WIDTH, new_head[1] % GRID_HEIGHT)
        elif not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            return False
        if not self.invincible and len(self.body) > 1 and new_head in self.body: return False
        self.body.insert(0, new_head)
        if self.grow:
            self.grow = False
        else:
            self.body.pop()
        return True

    def shrink(self, amount):
        for _ in range(amount):
            if len(self.body) > 1: self.body.pop()

    def draw(self, surface):
        color_head = INVINCIBLE_SNAKE_COLOR if self.invincible else SNAKE_HEAD_COLOR
        color_body = INVINCIBLE_SNAKE_COLOR if self.invincible else SNAKE_BODY_COLOR
        for i, seg in enumerate(self.body):
            pos = (seg[0] * GRID_SIZE + GRID_SIZE // 2, seg[1] * GRID_SIZE + GRID_SIZE // 2)
            if i == 0:
                pygame.draw.circle(surface, color_head, pos, GRID_SIZE // 2)
            else:
                pygame.draw.circle(surface, color_body, pos, GRID_SIZE // 2 - 1)


class Apple:
    def __init__(self, snake_body, surface_to_draw):
        self.surface = surface_to_draw
        self.randomize(snake_body)

    def randomize(self, snake_body):
        while True:
            self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if self.position not in snake_body: break

    def draw(self, surface):
        rect = pygame.Rect(self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        surface.blit(self.surface, rect)


class PowerUp:
    TYPES = {
        'golden': {'color': (255, 215, 0), 'char': 'G'},
        'star': {'color': (255, 255, 0), 'char': '★'},
        'poison': {'color': (148, 0, 211), 'char': 'P'}
    }

    def __init__(self, type, snake_body):
        self.type = type
        self.spawn_time = time.time()
        self.randomize(snake_body)

    def randomize(self, snake_body):
        while True:
            self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if self.position not in snake_body: break

    def draw(self, surface):
        info = self.TYPES[self.type]
        rect = pygame.Rect(self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, info['color'], rect, border_radius=5)
        char_text = FONT.render(info['char'], True, BLACK)
        surface.blit(char_text, char_text.get_rect(center=rect.center))


class Particle:
    def __init__(self, x, y, color):
        self.x, self.y, self.color = x, y, color
        self.vx, self.vy = random.uniform(-2, 2), random.uniform(-2, 2)
        self.life = 1.0

    def update_and_draw(self, surface, dt):
        self.x += self.vx;
        self.y += self.vy;
        self.life -= dt * 2
        if self.life > 0:
            radius = int(self.life * 10)
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), radius)
        return self.life > 0


# --- Helper Functions ---
def draw_grid(surface):
    grid_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for x in range(0, surface.get_width(), GRID_SIZE): pygame.draw.line(grid_surface, GRID_COLOR, (x, 0),
                                                                        (x, surface.get_height()))
    for y in range(0, surface.get_height(), GRID_SIZE): pygame.draw.line(grid_surface, GRID_COLOR, (0, y),
                                                                         (surface.get_width(), y))
    surface.blit(grid_surface, (0, 0))


def draw_control_zones(surface, zones, current_active_zone):
    for name, rect in zones.items():
        zone_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        color = ZONE_ACTIVE_COLOR if name == current_active_zone else ZONE_COLOR
        zone_surface.fill(color)
        surface.blit(zone_surface, rect.topleft)
        arrow_char = {'up': '▲', 'down': '▼', 'left': '◄', 'right': '►'}[name]
        arrow_text = ARROW_FONT.render(arrow_char, True, WHITE)
        surface.blit(arrow_text, arrow_text.get_rect(center=rect.center))


# --- Main Game ---
def main():
    global running, game_state, snake
    running = True
    with state_lock:
        game_state = 'intro'
    tracker_thread = threading.Thread(target=hand_tracking_worker, daemon=True);
    tracker_thread.start()
    apple_img_surface = create_apple_surface(GRID_SIZE)
    snake = Snake()
    apple = Apple(snake.body, apple_img_surface)
    powerups, particles = [], []
    score, highscore = 0, load_highscore()
    game_speed = 4.0
    apples_eaten_since_powerup, screen_shake, move_timer = 0, 0, 0
    zone_width, zone_height = WIDTH // 4, HEIGHT // 4
    zones = {
        'up': pygame.Rect(zone_width, 0, zone_width * 2, zone_height),
        'down': pygame.Rect(zone_width, HEIGHT - zone_height, zone_width * 2, zone_height),
        'left': pygame.Rect(0, zone_height, zone_width, zone_height * 2),
        'right': pygame.Rect(WIDTH - zone_width, zone_height, zone_width, zone_height * 2)
    }

    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
            if event.type == pygame.KEYDOWN and game_state == 'game_over' and event.key == pygame.K_r: main(); return

        current_game_state, current_active_zone = '', None
        with state_lock:
            current_game_state, current_active_zone = game_state, active_zone

        if current_game_state in ['playing', 'paused']:
            if snake.invincible and time.time() > snake.invincibility_end_time: snake.invincible = False
            powerups = [p for p in powerups if time.time() - p.spawn_time < 15]
            if current_game_state == 'playing':
                move_interval = 1.0 / game_speed
                move_timer += dt
                if move_timer >= move_interval:
                    move_timer = 0
                    if not snake.move():
                        if sound_game_over: sound_game_over.play()
                        screen_shake = 20
                        with state_lock:
                            game_state = 'game_over'
                        if score > highscore: save_highscore(score)
                if snake.body[0] == apple.position:
                    if sound_crunch: sound_crunch.play()
                    for _ in range(20): particles.append(
                        Particle(apple.position[0] * GRID_SIZE + GAME_AREA_RECT.left + GRID_SIZE // 2,
                                 apple.position[1] * GRID_SIZE + GAME_AREA_RECT.top + GRID_SIZE // 2, RED))
                    snake.grow_snake();
                    apple.randomize(snake.body);
                    score += 1;
                    apples_eaten_since_powerup += 1
                    if apples_eaten_since_powerup >= 5 and random.random() < 0.5:
                        apples_eaten_since_powerup = 0
                        powerups.append(PowerUp(random.choice(['golden', 'star', 'poison']), snake.body))
                for p in powerups[:]:
                    if snake.body[0] == p.position:
                        if sound_powerup: sound_powerup.play()
                        for _ in range(20): particles.append(
                            Particle(p.position[0] * GRID_SIZE + GAME_AREA_RECT.left + GRID_SIZE // 2,
                                     p.position[1] * GRID_SIZE + GAME_AREA_RECT.top + GRID_SIZE // 2,
                                     p.TYPES[p.type]['color']))
                        if p.type == 'golden':
                            score += 5
                        elif p.type == 'star':
                            snake.invincible, snake.invincibility_end_time = True, time.time() + 5
                        elif p.type == 'poison':
                            snake.shrink(3)
                        powerups.remove(p)

        render_offset = [0, 0]
        if screen_shake > 0: screen_shake -= 1; render_offset = [random.randint(-5, 5), random.randint(-5, 5)]
        with frame_lock:
            if latest_frame is not None:
                frame_surface = pygame.surfarray.make_surface(np.swapaxes(latest_frame, 0, 1))
                frame_surface = pygame.transform.scale(frame_surface, (WIDTH, HEIGHT))
                screen.blit(frame_surface, render_offset)
            else:
                screen.fill(BLACK)

        game_surface = screen.subsurface(GAME_AREA_RECT)
        draw_grid(game_surface)

        if current_game_state == 'intro':
            title_text = TITLE_FONT.render("GESTURE SNAKE", True, WHITE)
            instr_text = FONT.render("Show index finger to start. Open palm to pause.", True, WHITE)
            screen.blit(title_text, title_text.get_rect(center=(WIDTH / 2, HEIGHT / 3)))
            screen.blit(instr_text, instr_text.get_rect(center=(WIDTH / 2, HEIGHT / 2)))
        elif current_game_state == 'game_over':
            game_over_text = GAME_OVER_FONT.render("GAME OVER", True, RED)
            final_score_text = FONT.render(f"Final Score: {score}", True, WHITE)
            restart_text = FONT.render("Press 'R' to Restart", True, WHITE)
            screen.blit(game_over_text, game_over_text.get_rect(center=(WIDTH / 2, HEIGHT / 3)))
            screen.blit(final_score_text, final_score_text.get_rect(center=(WIDTH / 2, HEIGHT / 2)))
            screen.blit(restart_text, restart_text.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 60)))
        else:
            apple.draw(game_surface);
            [p.draw(game_surface) for p in powerups];
            snake.draw(game_surface)
            score_text = FONT.render(f"Score: {score}", True, WHITE)
            highscore_text = FONT.render(f"High Score: {highscore}", True, WHITE)
            quit_text = FONT.render("Press 'ESC' to Quit", True, WHITE)
            screen.blit(score_text, (20, HEIGHT - 35))
            screen.blit(highscore_text, (220, HEIGHT - 35))
            screen.blit(quit_text, quit_text.get_rect(bottomright=(WIDTH - 20, HEIGHT - 10)))
            if snake.invincible:
                timer = snake.invincibility_end_time - time.time()
                timer_text = FONT.render(f"Invincible: {max(0, timer):.1f}s", True, INVINCIBLE_SNAKE_COLOR)
                screen.blit(timer_text, timer_text.get_rect(midtop=(WIDTH / 2, 10)))
            if current_game_state == 'paused':
                pause_text = PAUSE_FONT.render("PAUSED", True, WHITE)
                screen.blit(pause_text, pause_text.get_rect(center=(WIDTH / 2, HEIGHT / 2)))

        draw_control_zones(screen, zones, current_active_zone)
        particles = [p for p in particles if p.update_and_draw(screen, dt)]
        pygame.display.flip()

    tracker_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
    pygame.quit()
    sys.exit()
