# COCONUT: Flappy Bird Co-Creative Edition

**Project Title:** COCONUT (Co-creating Co-creativity ‘N’ User Testing)
**Date:** 2026-01-06

A **Hybrid Neuro-Symbolic Co-Creative Game System** that collaborates with the player to design game levels in real-time. Built on *Flappy Bird*, this system acts as an intelligent design partner, negotiating physics, difficulty, and aesthetics after every gameplay session.

---

## 🚀 Key Features

*   **Turn-Based Co-Creation**: Use the "Death" moment as a design intervention. The AI suggests changes; you Negotiate.
*   **Hybrid Semantic Engine**:
    *   **Deterministic Physics**: Uses robust **Regex-Based Parsing** for precise control over Speed, Gravity, and Gaps.
    *   **Creative AI**: Uses **LLM Zero-Shot Classification** (`flan-t5`) to interpret abstract visual concepts (e.g., *"Make it spooky"* -> Hell Theme).
*   **Procedural Art Generation**: Dynamic asset generation for 12+ themes (Underwater, Matrix, Space, etc.) using parametric algorithms.
*   **Research Logging**: Automatically captures every user prompt and preference rating for analysis.

---

## 🎮 How to Play

### Controls
*   **SPACE**: Jump / Flap.
*   **M**: Modify the level (opens chat box during negotiation).
*   **1 / 2 / 3**: Rate the level (Good / Okay / Bad) after playing.

### The Loop
1.  **Play**: Fly as far as you can.
2.  **Crash**: The game pauses. The AI proposes a change (e.g., *"Try Neon Mode with Low Gravity"*).
3.  **Negotiate**:
    *   Press **SPACE** to accept the AI's idea.
    *   Press **M** to type your own idea (e.g., *"Make it faster with moving pipes"*).
4.  **Rate**: After the next round, rate the experience to teach the system.

---

## 🛠️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/flappy-bird-cocreative.git
    cd flappy-bird-cocreative
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Core Requirements: `pygame`, `numpy`, `transformers`, `torch`.*

3.  **Run the Game**:
    ```bash
    python main.py
    ```

---

## 🧠 System Architecture

The system follows a modular 4-layer architecture designed for research flexibility:

1.  **Game Engine (`world.py`, `pipe.py`)**: 
    *   Purely reactive physics engine.
    *   Supports variable gravity, speed, gap size, and **Moving Pipes**.
2.  **Creative State (`creative_state.py`)**: 
    *   Manages the "Creative Session". tracks history and persists feedback.
3.  **Intelligence Layer (`cocreative.py`)**: 
    *   **SemanticBrain**: The Hybrid Parser (Regex + LLM).
    *   **GenerativeArt**: The Procedural Renderer.
4.  **Interaction Layer (`main.py`)**: 
    *   Finite State Machine managing the `PLAY` <-> `NEGOTIATE` cycle.

---

## 🧪 Research & Evaluation

This system is designed for the **COCONUT** research project to evaluate:
*   User engagement with co-creative AI.
*   The effectiveness of hybrid (Deterministic + Generative) control systems.

Session data is logged to `session_feedback.json` for post-hoc analysis.

---

## 📜 Credits

*   **Lead Developer**: [Your Name/User]
*   **Framework**: Pygame
*   **AI Model**: Google Flan-T5 (HuggingFace)
