# cocreative.py
import os
import json
import random
import re
import threading
import queue
import uuid
from datetime import datetime

import pygame

try:
    import torch 
except Exception:
    torch = None

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


import sys

#Configuration
SESSION_ASSETS_DIR = os.path.join("assets", "session") 
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

CONFIDENCE_THRESHOLD = 0.65
PROPOSAL_CONFIDENCE_MIN = 0.60

PARSER_MAX_TOKENS = 64
PARSER_TEMPERATURE = 0.0 
PROPOSAL_MAX_TOKENS = 64
PROPOSAL_TEMPERATURE = 0.9

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.abspath(".")

SESSION_LOG_PATH = os.path.join(base_dir, "session_cocreative_log.json")

# predefined visuals
THEME_PRESETS = {
    "space":       {"sky": "space",      "ground": "tech",      "pipe": "tech"},
    "moon":        {"sky": "space",      "ground": "rock",      "pipe": "rock"},
    "hell":        {"sky": "hell",       "ground": "lava",      "pipe": "rock"},
    "snow":        {"sky": "snow",       "ground": "ice",       "pipe": "icy"},
    "ice":         {"sky": "snow",       "ground": "ice",       "pipe": "icy"},
    "desert":      {"sky": "desert",     "ground": "sand",      "pipe": "cactus"},
    "candy":       {"sky": "candy",      "ground": "chocolate", "pipe": "striped"},
    "matrix":      {"sky": "matrix",     "ground": "tech",      "pipe": "neon"},
    "night":       {"sky": "night",      "ground": "grass",     "pipe": "green"},
    "day":         {"sky": "day",        "ground": "grass",     "pipe": "green"},
    "underwater":  {"sky": "underwater", "ground": "sand",      "pipe": "green"},
    "forest":      {"sky": "forest",     "ground": "grass",     "pipe": "wood"},
    "sunset":      {"sky": "sunset",     "ground": "sand",      "pipe": "rust"},
}


# System Prompt for Player Prompt Parsing
PROMPT_PARSER_SYSTEM = """You are a precise intent parser for a Flappy Bird-like game.
Understand the player request and return ONLY valid JSON.

Possible actions:
- movement: "enable" / "disable" / "unchanged"
- speed: "faster" / "slower" / "unchanged"
- gravity: "heavier" / "lighter" / "unchanged"
- gap: "wider" / "narrower" / "unchanged"
- theme: "space","moon","hell","snow","desert","candy","matrix","night","day","underwater","forest","sunset","none"

Rules:
- "don't stop moving", "keep moving", "make pipes alive" -> enable
- "stop moving", "freeze pipes", "make static" -> disable
- Double negatives count as positive (don't disable = enable)
- Be conservative: use "unchanged" + low confidence if unclear
- Only change theme if clearly asked
- Output ONLY JSON, nothing else:

{
  "movement": "enable"|"disable"|"unchanged",
  "speed": "faster"|"slower"|"unchanged",
  "gravity": "heavier"|"lighter"|"unchanged",
  "gap": "wider"|"narrower"|"unchanged",
  "theme": "space"|...|"none",
  "confidence": 0.0-1.0
}
"""


# System Prompt for Creative Proposal Generation
PROPOSAL_SYSTEM = """You are a creative game mode suggester for a Flappy Bird-like game.
Given game stats, suggest a fun, exciting next level/mode name and style.
Make it sound appealing based on performance.
Examples: "Neon Space Dash", "Impossible Hell Run", "Floaty Moon Vibes"

Return ONLY JSON:
{
  "proposal_text": "short exciting string (max 8 words)",
  "confidence": 0.0-1.0
}
"""

class SessionLogger:
    def __init__(self, path=SESSION_LOG_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._entries = []
        self._index = {}  # level_id -> index
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._entries = data
                for i, e in enumerate(self._entries):
                    lid = e.get("level_id")
                    if isinstance(lid, str):
                        self._index[lid] = i
        except Exception:
            # if file is corrupted, don't crash the game
            self._entries = []
            self._index = {}

    def _atomic_save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def add_level_applied(
        self,
        *,
        prompt: str,
        source: str,
        intent: dict,
        params_before: dict,
        params_after: dict,
        applied: bool
    ) -> str:
        level_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"

        entry = {
            "level_id": level_id,
            "ts": now,
            "source": source if source else "unknown",
            "prompt": prompt,
            "intent": intent if isinstance(intent, dict) else {},
            "confidence": float((intent or {}).get("confidence", 0.0) or 0.0),
            "applied": bool(applied),
            "params_before": params_before,
            "params_after": params_after,
            "rating": None,
            "rating_ts": None,
        }

        with self._lock:
            self._entries.append(entry)
            self._index[level_id] = len(self._entries) - 1
            self._atomic_save()

        return level_id

    def set_rating(self, level_id: str, rating: str) -> bool:
        if rating not in ("good", "okay", "bad"):
            return False

        with self._lock:
            idx = self._index.get(level_id)
            if idx is None:
                return False
            self._entries[idx]["rating"] = rating
            self._entries[idx]["rating_ts"] = datetime.utcnow().isoformat() + "Z"
            self._atomic_save()

        return True


class LLMParser:
    def __init__(self):
        self.llm = None

        if not HAS_TRANSFORMERS:
            print("Check if transformers is installed...")
            return

        try:
            print(f"Loading {LLM_MODEL}... wait it takes a while for the first time run :)")

            try:
                self.llm = pipeline(
                    "text-generation",
                    model=LLM_MODEL,
                    device="cpu",
                    dtype="auto",
                )
            except TypeError:
                self.llm = pipeline(
                    "text-generation",
                    model=LLM_MODEL,
                    device="cpu",
                    torch_dtype="auto",
                )

            print("LLM loaded successfully.")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            self.llm = None

    def _messages_to_prompt(self, messages):
        try:
            tok = getattr(self.llm, "tokenizer", None)
            if tok is not None and hasattr(tok, "apply_chat_template"):
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        except Exception:
            pass

        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role.upper()}]\n{content}")
        parts.append("[ASSISTANT]\n")
        return "\n\n".join(parts)

    def _normalize_to_text(self, response):
        if isinstance(response, list) and response:
            first = response[0]
            if isinstance(first, dict):
                text = first.get("generated_text") or first.get("text")
            else:
                text = first

            if isinstance(text, list):
                # Sometimes returns chat-shaped list. Take last assistant content.
                for msg in reversed(text):
                    if isinstance(msg, dict) and "content" in msg:
                        return str(msg["content"])
                return json.dumps(text)

            return str(text)

        if isinstance(response, dict):
            return str(response.get("generated_text") or response.get("text") or "")

        if isinstance(response, str):
            return response

        return ""

    def _extract_json(self, text):
        if not isinstance(text, str):
            return None

        matches = re.findall(r"\{[\s\S]*?\}", text)
        if not matches:
            return None

        for candidate in reversed(matches):
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    def _postprocess_intent(self, parsed):
        if not isinstance(parsed, dict):
            return {"confidence": 0.0}

        out = {
            "movement": parsed.get("movement", "unchanged"),
            "speed": parsed.get("speed", "unchanged"),
            "gravity": parsed.get("gravity", "unchanged"),
            "gap": parsed.get("gap", "unchanged"),
            "theme": parsed.get("theme", "none"),
            "confidence": parsed.get("confidence", 0.0),
        }

        if out["movement"] not in ("enable", "disable", "unchanged"):
            out["movement"] = "unchanged"
        if out["speed"] not in ("faster", "slower", "unchanged"):
            out["speed"] = "unchanged"
        if out["gravity"] not in ("heavier", "lighter", "unchanged"):
            out["gravity"] = "unchanged"
        if out["gap"] not in ("wider", "narrower", "unchanged"):
            out["gap"] = "unchanged"
        if out["theme"] not in set(THEME_PRESETS.keys()) | {"none"}:
            out["theme"] = "none"

        try:
            out["confidence"] = float(out["confidence"])
        except Exception:
            out["confidence"] = 0.0
        out["confidence"] = max(0.0, min(1.0, out["confidence"]))

        return out

    def _call_llm(self, system_prompt, user_message, max_tokens, temperature):
        if not self.llm:
            return {"confidence": 0.0}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]

        prompt = self._messages_to_prompt(messages)

        try:
            response = self.llm(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                return_full_text=False,
            )
        except Exception as e:
            print(f"LLM error: {e}")
            return {"confidence": 0.0}

        text = self._normalize_to_text(response)
        parsed = self._extract_json(text)
        if parsed is None:
            print("No valid JSON found in LLM output")
            return {"confidence": 0.0}

        return parsed

    def parse_prompt(self, user_text):
        parsed = self._call_llm(
            PROMPT_PARSER_SYSTEM,
            f"Player request: {user_text}",
            max_tokens=PARSER_MAX_TOKENS,
            temperature=PARSER_TEMPERATURE
        )
        return self._postprocess_intent(parsed)

    def generate_proposal(self, score, time_alive, performance_snapshot=None, good_examples=None):
        status = f"Score: {score}\nTime survived: {time_alive} seconds"
        if performance_snapshot:
            status += f"\nPerformance: {performance_snapshot}"
        if good_examples:
            status += "\nGood previous examples:\n" + "\n".join([f"- {ex}" for ex in good_examples[:3]])

        parsed = self._call_llm(
            PROPOSAL_SYSTEM,
            status,
            max_tokens=PROPOSAL_MAX_TOKENS,
            temperature=PROPOSAL_TEMPERATURE
        )

        if isinstance(parsed, dict):
            try:
                conf = float(parsed.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            text = parsed.get("proposal_text", "") or ""

            if conf >= PROPOSAL_CONFIDENCE_MIN and isinstance(text, str) and len(text.strip()) >= 3:
                return text.strip()

        return "Try something new!"


class GenerativeArt:
    def __init__(self):
        self.surface = None

    def create_theme_surfaces(self, sky="day", ground="grass", pipe="green"):
        bg = self._generate_bg(sky)
        gr = self._generate_ground(ground)
        pp = self._generate_pipe(pipe)
        return bg, gr, pp

    def _generate_bg(self, style):
        if style == "space":
            surf = self._draw_gradient((0, 0, 20), (10, 10, 60))
            self._add_stars(surf, 100)
            return surf
        if style == "hell":
            surf = self._draw_gradient((60, 0, 0), (20, 0, 0))
            self._add_mountains(surf, (40, 0, 0))
            return surf
        if style == "snow":
            surf = self._draw_gradient((200, 240, 255), (255, 255, 255))
            self._add_stars(surf, 30)
            return surf
        if style == "desert":
            return self._draw_gradient((255, 150, 50), (255, 220, 100))
        if style == "night":
            surf = self._draw_gradient((0, 0, 40), (20, 20, 80))
            self._add_stars(surf, 50)
            return surf
        if style == "matrix":
            surf = pygame.Surface((512, 384))
            surf.fill((0, 0, 0))
            self._add_matrix_rain(surf)
            return surf
        if style == "underwater":
            surf = self._draw_gradient((0, 50, 150), (0, 150, 200))
            self._add_bubbles(surf, 40)
            return surf
        if style == "forest":
            surf = self._draw_gradient((50, 150, 50), (150, 220, 150))
            self._add_trees(surf, 10)
            return surf
        if style == "sunset":
            surf = self._draw_gradient((100, 50, 100), (255, 150, 50))
            self._add_sun(surf)
            return surf

        surf = self._draw_gradient((100, 200, 255), (200, 230, 255))
        self._add_clouds(surf, 5)
        return surf

    def _generate_ground(self, style):
        surf = pygame.Surface((512, 96))
        if style == "ice":
            surf.fill((220, 240, 255))
        elif style == "lava":
            surf.fill((80, 0, 0))
        elif style == "tech":
            surf.fill((0, 50, 0))
            pygame.draw.line(surf, (200, 255, 200), (0, 48), (512, 48), 2)
        elif style == "sand":
            surf.fill((240, 230, 140))
        else:
            surf.fill((100, 200, 100))
        return surf

    def _generate_pipe(self, style):
        c = (50, 200, 50)
        if style == "icy":
            c = (150, 200, 255)
        elif style == "rock":
            c = (80, 80, 80)
        elif style == "rust":
            c = (120, 80, 50)
        elif style == "neon":
            c = (0, 255, 255)
        elif style == "wood":
            c = (100, 60, 30)
        elif style == "cactus":
            c = (50, 160, 50)
        elif style == "striped":
            c = (200, 80, 160)
        elif style == "green":
            c = (50, 200, 50)

        surf = pygame.Surface((64, 384), pygame.SRCALPHA)
        surf.fill(c)
        pygame.draw.rect(surf, (0, 0, 0), (0, 0, 64, 384), 4)

        if style == "neon":
            pygame.draw.rect(surf, (255, 255, 255), (10, 0, 10, 384))
        if style == "wood":
            pygame.draw.rect(surf, (90, 50, 20), (15, 0, 5, 384))
            pygame.draw.rect(surf, (90, 50, 20), (45, 0, 5, 384))

        return surf

    def _draw_gradient(self, c1, c2):
        surf = pygame.Surface((512, 384))
        for y in range(384):
            t = y / 384
            r = c1[0] + (c2[0] - c1[0]) * t
            g = c1[1] + (c2[1] - c1[1]) * t
            b = c1[2] + (c2[2] - c1[2]) * t
            pygame.draw.line(surf, (int(r), int(g), int(b)), (0, y), (512, y))
        return surf

    def _add_stars(self, surf, n):
        for _ in range(n):
            pygame.draw.circle(surf, (255, 255, 255), (random.randint(0, 512), random.randint(0, 384)), 1)

    def _add_clouds(self, surf, n):
        for _ in range(n):
            pygame.draw.circle(surf, (255, 255, 255), (random.randint(0, 512), random.randint(0, 384)), 20)

    def _add_bubbles(self, surf, n):
        for _ in range(n):
            pygame.draw.circle(
                surf, (255, 255, 255),
                (random.randint(0, 512), random.randint(0, 384)),
                random.randint(2, 6), 1
            )

    def _add_trees(self, surf, n):
        for _ in range(n):
            x = random.randint(0, 512)
            pygame.draw.polygon(
                surf, (30, 100, 30),
                [(x, 384), (x + 30, 384), (x + 15, 384 - random.randint(50, 150))]
            )

    def _add_sun(self, surf):
        pygame.draw.circle(surf, (255, 200, 50), (256, 384), 100)

    def _add_mountains(self, surf, color):
        pygame.draw.polygon(surf, color, [(0, 384), (100, 250), (200, 384), (300, 200), (512, 384)])

    def _add_matrix_rain(self, surf):
        for _ in range(50):
            x, h = random.randint(0, 512), random.randint(5, 30)
            pygame.draw.rect(surf, (0, 255, 0), (x, random.randint(0, 384), 2, h))


class CoCreativeManager:
    """
    LLM decides intent. Main loop never blocks.
    - request_prompt(text): schedules parse
    - request_proposal(...): schedules proposal
    - poll(): retrieves finished jobs
    - apply_intent(...): applies intent quickly on main thread
    """
    def __init__(self, theme_manager, log_path=SESSION_LOG_PATH):
        self.theme_manager = theme_manager

        self.parser = LLMParser()
        self.ai_art = GenerativeArt()
        self.logger = SessionLogger(path=log_path)

        self._in_q = queue.Queue()
        self._out_q = queue.Queue()
        self.ai_thinking = False

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        while True:
            job = self._in_q.get()
            try:
                kind = job["kind"]

                if kind == "parse":
                    text = job["text"]
                    intent = self.parser.parse_prompt(text)
                    self._out_q.put({"kind": "parse_done", "text": text, "intent": intent})

                elif kind == "proposal":
                    score = job["score"]
                    time_alive = job["time_alive"]
                    snapshot = job.get("snapshot")
                    good_examples = job.get("good_examples")
                    proposal = self.parser.generate_proposal(score, time_alive, snapshot, good_examples)
                    self._out_q.put({"kind": "proposal_done", "proposal": proposal})

            except Exception as e:
                self._out_q.put({"kind": "error", "error": str(e)})
            finally:
                self._in_q.task_done()

    def request_prompt(self, text, source="user"):
        if not text or not text.strip():
            return
        self.ai_thinking = True
        self._in_q.put({"kind": "parse", "text": text, "source": source})

    def request_proposal(self, score, time_alive, snapshot=None, good_examples=None):
        self.ai_thinking = True
        self._in_q.put({
            "kind": "proposal",
            "score": score,
            "time_alive": time_alive,
            "snapshot": snapshot,
            "good_examples": good_examples,
        })

    def poll(self):
        msgs = []
        try:
            while True:
                msgs.append(self._out_q.get_nowait())
        except queue.Empty:
            pass

        if msgs:
            # if anything arrived, we are not "thinking" anymore
            self.ai_thinking = False
        return msgs

    def record_feedback(self, rating: str, level_id: str = None):
        """
        Call this from main.py when user presses 1/2/3.
        If level_id is not provided, we attach to the last applied level.
        """
        lid = level_id or self.last_level_id
        if not lid:
            print("Warning: record_feedback called but no level_id is available.")
            return False

        ok = self.logger.set_rating(lid, rating)
        if not ok:
            print(f"Warning: could not set rating='{rating}' for level_id={lid}")
        return ok

    def apply_intent(self, text, intent, current_params, source="unknown"):
        params_before = dict(current_params)  # shallow copy is enough for numbers/strings
        conf = float((intent or {}).get("confidence", 0.0) or 0.0)

        print(f"\nApplying LLM intent for prompt: '{text}' (conf={conf:.2f})")

        # If low confidence: log it (applied=False) but do not change params
        if conf < CONFIDENCE_THRESHOLD:
            print(f"Confidence too low ({conf:.2f}) - no change")
            self.last_level_id = self.logger.add_level_applied(
                prompt=text,
                source=source,
                intent=intent if isinstance(intent, dict) else {"confidence": conf},
                params_before=params_before,
                params_after=params_before,
                applied=False,
            )
            return current_params

        p = dict(current_params)

        mov = intent.get("movement", "unchanged")
        if mov == "enable":
            p["pipe_move_speed"] = 1.0
            print("Enabled moving pipes")
        elif mov == "disable":
            p["pipe_move_speed"] = 0.0
            print("Disabled moving pipes")

        spd = intent.get("speed", "unchanged")
        if spd == "faster":
            p["speed"] = max(p.get("speed", -6.0) - 3.0, -15.0)
            print("Faster")
        elif spd == "slower":
            p["speed"] = min(p.get("speed", -6.0) + 3.0, -2.0)
            print("Slower")

        grav = intent.get("gravity", "unchanged")
        if grav == "heavier":
            p["gravity"] = min(p.get("gravity", 0.5) + 0.2, 1.3)
            print("Heavier")
        elif grav == "lighter":
            p["gravity"] = max(p.get("gravity", 0.5) - 0.2, 0.1)
            print("Lighter")

        gap = intent.get("gap", "unchanged")
        if gap == "wider":
            p["gap_multiplier"] = min(p.get("gap_multiplier", 1.0) + 0.3, 2.5)
            print("Wider gap")
        elif gap == "narrower":
            p["gap_multiplier"] = max(p.get("gap_multiplier", 1.0) - 0.3, 0.5)
            print("Narrower gap")

        old_theme = (p.get("sky_style"), p.get("ground_style"), p.get("pipe_style"))

        theme = intent.get("theme", "none")
        if theme in THEME_PRESETS:
            preset = THEME_PRESETS[theme]
            p["sky_style"] = preset.get("sky", p.get("sky_style", "day"))
            p["ground_style"] = preset.get("ground", p.get("ground_style", "grass"))
            p["pipe_style"] = preset.get("pipe", p.get("pipe_style", "green"))
            print(f"Theme: {theme}")

        new_theme = (p.get("sky_style"), p.get("ground_style"), p.get("pipe_style"))
        theme_changed = (new_theme != old_theme)

        # physics
        g = p.get("gravity", 0.5)
        if g < 0.25:
            p["vel"] = 7.0
        elif g < 0.4:
            p["vel"] = 8.0
        elif g > 0.9:
            p["vel"] = 14.0
        elif g > 0.7:
            p["vel"] = 12.0
        else:
            p["vel"] = 10.0

        print(
            f"Final Speed:{p.get('speed','?'):.1f} "
            f"Grav:{p.get('gravity','?'):.1f} "
            f"Move:{p.get('pipe_move_speed','?')}"
        )

        if theme_changed:
            self._apply_theme_from_styles(
                sky=p.get("sky_style", "day"),
                ground=p.get("ground_style", "grass"),
                pipe=p.get("pipe_style", "green"),
            )

        self.last_level_id = self.logger.add_level_applied(
            prompt=text,
            source=source,
            intent=intent if isinstance(intent, dict) else {},
            params_before=params_before,
            params_after=p,
            applied=True,
        )

        return p

    def _apply_theme_from_styles(self, sky, ground, pipe):
        bg_s, gr_s, pp_s = self.ai_art.create_theme_surfaces(sky=sky, ground=ground, pipe=pipe)

        try:
            self.theme_manager.themes["custom"] = {
                "background": bg_s.convert(),
                "ground": gr_s.convert(),
                "pipe": pp_s.convert_alpha(),
            }
            self.theme_manager.set_theme("custom")
        except Exception as e:
            print(f"Theme apply error: {e}")
