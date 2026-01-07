# COCONUT: Flappy Bird Co-Creative Edition

**Project Title:** Wings of Co-Creation: Human-AI Level Design Through Play

> A **Turn-Based Co-Creative Game System** in Flappy Bird that partners with you to design levels in real-time. It negotiates physics, difficulty, and aesthetics with an AI designer after every run.

---

## 🚀 Key Features

*   **Turn-Based Co-Creation**: The "Death" moment is a design intervention. The AI analyzes your play and suggests improvements.
*   **Advanced AI**:
    *   **LLM Intelligence**: Uses **Qwen2.5-1.5B-Instruct** for deep semantic understanding of complex requests (e.g., *"Make it a floaty moon level"*).
*   **Dynamic Visuals & Physics**:
    *   **Procedural Themes**: 12+ styles (Neon, Hell, Matrix, Snow, etc.) generated algorithmically.
    *   **Variable Physics**: Real-time control over Gravity, Speed, Gap Size, and Moving Pipes.
*   **Logging**:
    *   **User Feedback**: Rate levels (Good/Bad) to save preferences (this was used for evaluation of the game).
    *   **Audit Trails**: Every prompt and AI decision is logged to `session_cocreative_log.json` for analysis.

---

## 🎮 How to Play

### Controls
*   **SPACE**: Jump / Flap.
*   **SHIFT**: Open the Design Chat (anytime).
*   **M**: Modify the AI's proposal during negotiation.
*   **1 / 2 / 3**: Rate the level (Good / Okay / Bad) after dying.

### The Loop
1.  **Play**: Fly as far as you can.
2.  **Crash**: The game pauses. The AI analyzes your score.
3.  **Proposal**: The AI suggests a change (e.g., *"You're doing great! Try 'Abyssal Void' mode with moving pipes"*).
4.  **Negotiate**:
    *   Press **Y** to accept.
    *   Press **N** to reject.
    *   Press **M** to modify (e.g., *"Make it faster instead"*).
5.  **Rate**: Teach the system what is fun by rating the level after playing it.

---

## 🛠️ Installation & Setup

### Running from Source

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/deepaksomesh/flappy-bird-cocreative.git
    cd flappy-bird-cocreative
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Key packages: `pygame`, `numpy`, `transformers`, `torch`, `pyinstaller`.*
    
    If any issues with pygame installation, use the below to install it separately
    ```bash
    python -m pip install pygame
    pip install numpy transformers torch pyinstaller
    ```

4.  **Run the Game**:
    ```bash
    python main.py
    ```

---

## 📦 Building form Source
To generate your own `.exe` file:

```bash
python build_exe.py
```
*   This uses `PyInstaller` to bundle the Python interpreter, dependencies, and assets into a single file.
*   It generates **`FlappyBirdCoCreative.exe`** in a `\dist` folder (This was then shared across different users to test the game).
Note*: The first launch of the game may take 10-20 seconds to initialize the AI engine and to start the game.

To avoid all the technical setup, use this link to download the **`FlappyBirdCoCreative.exe`** [Drive Link](https://drive.google.com/file/d/1iYmmdmT6Mdx4QsaMuj8DXjxtgOxGbnIn/view?usp=drive_link)

---

## 🧠 System Architecture

High-Level Architecture:

| Layer | Component | Description |
| :--- | :--- | :--- |
| **1. UI & State** | `main.py` | Manages the Game Loop and State Machine (`WAITING`, `PLAYING`, `NEGOTIATING`). Handles an input box and overlay rendering. |
| **2. Engine** | `world.py` | Reactive physics engine. Handles collisions, scoring, etc. |
| **3. Memory** | `creative_state.py` | Persists session history and user feedback to `session_feedback.json`. |
| **4. Intelligence** | `cocreative.py` | The "Brain". Runs the **Qwen LLM** in a background worker thread to parse natural language and generate level parameters without blocking the main UI thread. |

---

## 📝 Logging & Data
The system automatically creates two log files next to the executable (or script):
*   **`session_feedback.json`**: Stores your ratings (Good/Bad) for specific levels.
*   **`session_cocreative_log.json`**: A detailed technical log of every AI interaction, prompt confidence score, and applied parameter change.

---

## Contributors
* Deepak Somesh K J
* Nallathambi V
* Ekansh
* Aparajita

---

## 📜 Credits
*   **Framework**: Pygame
*   **AI Model**: Qwen2.5-1.5B-Instruct (Alibaba Cloud)
*   **Original Game**: Flappy Bird (Dong Nguyen)
