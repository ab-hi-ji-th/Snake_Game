# 🐍 Gesture Snake: A Modern Twist on a Classic 🎮

> A classic Snake game, reimagined for the modern era. Control the snake in real-time using nothing but your hand gestures, captured directly from your webcam!

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Pygame](https://img.shields.io/badge/Pygame-2.5.2-green?style=for-the-badge&logo=pygame)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-blue?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

---

### 🎬 Gameplay Demo

This project is not just a game; it's an interactive experience. Instead of old-school keyboard controls, it uses **OpenCV** for video capture and **Google's MediaPipe** framework to perform real-time hand landmark detection, translating your gestures into commands.

***(➡️ PASTE YOUR GAMEPLAY GIF/VIDEO HERE ⬅️)***

![Gameplay Demo GIF](<path_to_your_gameplay_demo.gif>)

---

### ✨ Key Features

* **Gesture-Based Controls:** Navigate the snake using just your index finger.
* **Live Webcam Background:** Your own room becomes the game's dynamic background.
* **Pause/Unpause Gesture:** Show an open palm to pause the action and a pointing finger to resume.
* **Dynamic Power-ups:**
    * ⭐ **Invincibility Star:** Become invincible for a short period, passing through walls and yourself!
    * 💰 **Golden Apple:** Boost your score with this rare item.
    * ☠️ **Poison Apple:** Watch out! This will shrink your snake.
* **Polished UX/UI:**
    * Fullscreen borderless window for full immersion.
    * On-screen UI for score, high score, and power-up timers.
    * Particle effects and screen shake animations for satisfying feedback.
* **Sound Effects:** Audio feedback for eating apples, collecting power-ups, and game over.
* **Persistent High Score:** Your best score is saved locally and displayed, challenging you to beat it every time!

---

### 📸 Screenshots

| Home Screen                               | Playing                                    |
| ------------------------------------------ | ------------------------------------------ |
| **(![1000137740](https://github.com/user-attachments/assets/8a91c166-8b6f-49e7-9c16-8427c5ef9a08)
)** <br> ![Image 1](<path_to_your_image_1.png>) | **(![1000137742](https://github.com/user-attachments/assets/226102b6-bdf4-4371-95d5-5191e945c383)
)** <br> ![Image 2](<path_to_your_image_2.png>) |
| **Paused** | **Game Over** |
| **(![1000137741](https://github.com/user-attachments/assets/77dcfe88-ddc3-4afa-ad0e-c89ea1fde4d1)
)** <br> ![Image 3](<path_to_your_image_3.png>) | **(![1000137749](https://github.com/user-attachments/assets/44688f4e-a6b8-4955-bb4c-2636958e58fb)
)** <br> ![Image 4](<path_to_your_image_4.png>) |

---

### 💻 Tech Stack

This project was built using the following technologies:

* **Python:** The core programming language.
* **Pygame:** For creating the game window, drawing elements, and handling sound.
* **OpenCV:** To capture the live video feed from the webcam.
* **Google MediaPipe:** For real-time, high-fidelity hand tracking and gesture recognition.
* **NumPy:** For efficient numerical operations on video frames.

---

### 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

**Prerequisites:**
* Python 3.8+
* A webcam

**Installation:**

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/ab-hi-ji-th/Snake_Game.git](https://github.com/ab-hi-ji-th/Snake_Game.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd Snake_Game
    ```
3.  **Create and activate a virtual environment:**
    * On Windows:
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
4.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```
5.  **Run the game!**
    ```sh
    python snake_game.py
    ```

---

### 🎮 How to Play

* **Start Game:** Show your **index finger** to the camera when the intro screen appears.
* **Control Snake:** Move your index finger into the highlighted zones on the screen to direct the snake.
* **Pause Game:** Show an **open palm** (all fingers extended).
* **Resume Game:** Show your **index finger** again.
* **Quit Game:** Press the `ESC` key at any time.

---

### 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

### 💬 Contact

Abhijith P V - [[LinkedIn](https://www.linkedin.com/in/abhijith-p-v-74bb6a281/)] - [abhijthpv32@gmail.com](mailto:abhijthpv32@gmail.com)

Project Link: [https://github.com/ab-hi-ji-th/Snake_Game](https://github.com/ab-hi-ji-th/Snake_Game)
