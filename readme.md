# COCONUT: Flappy Bird Co-Creative Edition

**Project Title:** COCONUT (Co-creating Co-creativity ‘N’ User Testing)
**Version:** 2.0 (Distribution Ready)

> A **Turn-Based Co-Creative Game System** that partners with you to design levels in real-time. Negotiate physics, difficulty, and aesthetics with an advanced AI designer after every run.

---

## 🚀 Key Features

*   **Turn-Based Co-Creation**: The "Death" moment is a design intervention. The AI analyzes your play and suggests improvements.
*   **Advanced AI (Deep Thinking)**:
    *   **Pure LLM Intelligence**: Powered by **Qwen2.5-1.5B-Instruct** for deep semantic understanding of complex requests (e.g., *"Make it a floaty moon level"*).
    *   **Asynchronous Processing**: AI runs on a background thread, ensuring the game never freezes even while the LLM is "thinking."
*   **Dynamic Visuals & Physics**:
    *   **Procedural Themes**: 12+ styles (Neon, Hell, Matrix, Snow, etc.) generated algorithmically.
    *   **Variable Physics**: Real-time control over Gravity, Speed, Gap Size, and **Moving Pipes**.
*   **Research-Grade Logging**:
    *   **User Feedback**: Rate levels (Good/Bad) to save preferences.
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

### Option A: Standalone Executable (Recommended for Players)
No coding knowledge required!
1.  Navigate to the `dist` folder.
2.  Run **`FlappyBirdCoCreative.exe`**.
3.  *Note*: The first launch may take 10-20 seconds to initialize the AI engine.

### Option B: Running from Source (Recommended for Developers)

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/flappy-bird-cocreative.git
    cd flappy-bird-cocreative
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Key packages: `pygame`, `numpy`, `transformers`, `torch`, `pyinstaller`.*

3.  **Run the Game**:
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
*   The output will be in the `dist/` folder.

---

## 🧠 System Architecture

The game uses a modern **4-Layer Asynchronous Architecture**:

| Layer | Component | Description |
| :--- | :--- | :--- |
| **1. UI & State** | `main.py` | Manages the Game Loop and State Machine (`WAITING`, `PLAYING`, `NEGOTIATING`). Handles the new smart input box and overlay rendering. |
| **2. Engine** | `world.py` | Reactive physics engine. Handles collisions, scoring, and the mathematical logic for **Moving Pipes**. |
| **3. Memory** | `creative_state.py` | Persists session history and user feedback to `session_feedback.json`. |
| **4. Intelligence** | `cocreative.py` | The "Brain". Runs the **Qwen LLM** in a background worker thread to parse natural language and generate level parameters without blocking the main UI thread. |

---

## 📝 Logging & Data
The system automatically creates two log files next to the executable (or script):
*   **`session_feedback.json`**: Stores your ratings (Good/Bad) for specific levels.
*   **`session_cocreative_log.json`**: A detailed technical log of every AI interaction, prompt confidence score, and applied parameter change.

---

## 📜 Credits
*   **Framework**: Pygame
*   **AI Model**: Qwen2.5-1.5B-Instruct (Alibaba Cloud)
*   **Original Game**: Flappy Bird (Dong Nguyen)
