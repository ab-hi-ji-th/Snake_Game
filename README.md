# 🐍 Gesture Snake: A Modern Twist on a Classic 🎮

> A classic Snake game, reimagined for the modern era. Control the snake in real-time using nothing but your hand gestures, captured directly from your webcam!

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Pygame](https://img.shields.io/badge/Pygame-2.5.2-green?style=for-the-badge&logo=pygame)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-blue?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

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

| Home Screen | Playing |
| :---: | :---: |
| <img src="<![1000137740](https://github.com/user-attachments/assets/6c0d3cbd-bc64-403b-8878-b0652d0203f9)
>" alt="Home Screen" width="400"> | <img src="<![1000137742](https://github.com/user-attachments/assets/eb9ccf79-79dc-4d24-b2ed-36d67be94047)
>" alt="Playing" width="400"> |
| **Paused** | **Game Over** |
| <img src="<![1000137741](https://github.com/user-attachments/assets/544fc2f7-d0b8-402e-81de-c1aadc897b35)
>" alt="Paused" width="400"> | <img src="<![1000137749](https://github.com/user-attachments/assets/819d197b-9af3-4832-9616-d7538df7ed9e)
>" alt="Game Over" width="400"> |

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
