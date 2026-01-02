import os
import pygame
import json
import random
import torch
import torch.nn as nn
import numpy as np
from io import BytesIO

# ==========================================
# 1. LOCAL LLM BRAIN (Semantic Understanding)
# ==========================================
class SemanticBrain:
    def __init__(self):
        self.pipe = None
        try:
            from transformers import pipeline
            print("Loading Local LLM (google/flan-t5-small)... this may take a moment...")
            # text2text-generation is perfect for "Translation" tasks
            self.pipe = pipeline("text2text-generation", model="google/flan-t5-small", device="cpu")
            print("LLM Loaded Successfully.")
        except ImportError:
            print("CRITICAL: 'transformers' library not found. Please run: pip install transformers sentencepiece")
            self.pipe = None
        except Exception as e:
            print(f"LLM Load Error: {e}")
            self.pipe = None

    def analyze(self, prompt, current_params):
        if not self.pipe:
            print("LLM not active. Returning default.")
            return current_params

        # Prompt Engineering for Flan-T5: Ask for Numerical Scales (1-10)
    def analyze(self, prompt, current_params):
        if not self.pipe:
            print("LLM not active. Returning default.")
            return current_params

    def analyze(self, prompt, current_params):
        if not self.pipe:
            print("LLM not active. Returning default.")
            return current_params

        # Compositional Prompt (Granular Control, less verbose)
        llm_prompt = (
            f"Classify '{prompt}'. "
            "Sky: (Day, Night, Space, Hell, Snow, Desert, Candy, Matrix, Industrial)? "
            "Ground: (Grass, Ice, Sand, Lava, Rock, Tech, Chocolate, Road)? "
            "Pipe: (Green, Icy, Cactus, Rock, Tech, Striped, Rusty, Neon)? "
            "Physics: Speed(Fast/Normal/Slow), Gravity(High/Low/Normal), Diff(Hard/Easy)?"
        )
        
        try:
            # Inference
            output = self.pipe(llm_prompt, max_new_tokens=96, do_sample=True, temperature=0.5)[0]['generated_text']
            print(f"LLM RAW OUTPUT: {output}")
            
            # Parse
            params = self._parse_llm_output(output, current_params)
            return params
        except Exception as e:
            print(f"LLM Inference Error: {e}")
            return current_params

    def _parse_llm_output(self, text, current_params):
        p = current_params.copy()
        text = text.lower()
        
        # Helper to find category
        def get_category(options, default):
            for opt in options:
                # Check for explicit word boundaries or specific phrasing
                if opt in text:
                    return opt
            return default

        # --- 1. Physics Mapping (Sturdier) ---
        # Explicit checks to avoid "Speed" keyword triggering logic
        is_fast = "speed=fast" in text or "fast" in text.replace("speed(fast", "") 
        is_slow = "speed=slow" in text or "slow" in text.replace("speed(fast/normal/slow)", "")
        
        is_heavy = "gravity=high" in text or "heavy" in text
        is_floaty = "gravity=low" in text or "float" in text
        
        # Speed & Distance
        if is_fast: 
             p['speed'] = -9.0
             p['dist'] = 6
        elif is_slow: 
             p['speed'] = -4.0
             p['dist'] = 4
        else: # Normal or implicit
             if "speed=normal" in text:
                 p['speed'] = -6.0
                 p['dist'] = 4

        # Gravity & Jump
        if is_heavy: 
            p['gravity'] = 0.8
            p['vel'] = 12.0
        elif is_floaty: 
            p['gravity'] = 0.25
            p['vel'] = 8.0
        else:
            if "gravity=normal" in text:
                p['gravity'] = 0.5
                p['vel'] = 10.0
            
        # Pattern
        if "diff=hard" in text or "hard" in text: 
            p['pattern'] = 'random' 
            p['gap_multiplier'] = 0.9
        elif "diff=easy" in text or "easy" in text: 
            p['pattern'] = 'ascending'
            p['gap_multiplier'] = 1.3
        else:
             if "diff=normal" in text:
                 p['pattern'] = 'zigzag'
                 p['gap_multiplier'] = 1.0
            
        # --- 2. Visual Composition (Granular) ---
        p['sky_style'] = get_category(['space', 'hell', 'snow', 'desert', 'candy', 'matrix', 'industrial', 'night'], 'day')
        p['ground_style'] = get_category(['ice', 'sand', 'lava', 'rock', 'tech', 'chocolate', 'road'], 'grass')
        p['pipe_style'] = get_category(['icy', 'cactus', 'rock', 'tech', 'striped', 'rusty', 'neon'], 'green')
        
        # Color override (optional)
        def get_val(keys):
            if isinstance(keys, str): keys = [keys]
            for k in keys:
                if k in text:
                    parts = text.split(k)
                    if len(parts) > 1: return parts[1].split(',')[0].strip()
            return ""
        p['llm_color'] = get_val(["color", "color:"])
        
        print(f"LLM Parsed: Speed={p.get('speed')} Grav={p.get('gravity')} -> Vis: {p['sky_style']}/{p['pipe_style']}")
        return p

    # --- 4. AI Creativity (Proposal Generation) ---
    def propose_change(self, score, time_alive):
        if not self.pipe: return "Make it easier"
        
        # Construct Observation Prompt
        status = "struggling" if score < 2 else "doing well" if score < 10 else "mastering it"
        
        # Randomize examples to prevent overfitting/repetition
        examples = [
            "'Make it fast and neon'", "'Go to space'", "'Make it slow and floaty'", 
            "'Ice world with high gravity'", "'Desert theme'", "'Matrix mode with tech pipes'"
        ]
        random.shuffle(examples)
        ex_str = ", ".join(examples[:3])

        llm_prompt = (
            f"Player is {status}. Score: {score}. "
            f"Suggest a creative game change. Examples: {ex_str}. "
            "Proposal: "
        )
        
        try:
            # Inference with repetition_penalty to stop loops
            output = self.pipe(llm_prompt, max_new_tokens=32, do_sample=True, temperature=0.9, repetition_penalty=1.2)[0]['generated_text']
            # Clean up output
            suggestion = output.split("Proposal:")[-1].strip().strip('"').strip("'")
            if len(suggestion) < 5: suggestion = "Surprise me!"
            
            print(f"AI Observation: Score={score} ({status}) -> Proposal: '{suggestion}'")
            return suggestion
        except Exception as e:
            print(f"LLM Reasoning Error: {e}")
            return "Make it easier"

# ==========================================
# 2. GENERATIVE ART ENGINE (Geometric)
# ==========================================
class GenerativeArt:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.asset_lib_path = os.path.join("assets", "layout")

    def create_theme(self, prompt, sky='day', ground='grass', pipe='green', color=None):
        print(f"Generative Art: Composing '{prompt}' -> Sky:{sky}, Ground:{ground}, Pipe:{pipe}, Tint:{color}")
        
        # 1. Color Selection
        if color:
            colors = {
                "red": (200, 50, 50), "green": (50, 200, 50), "blue": (50, 100, 255),
                "yellow": (200, 200, 50), "gray": (120, 120, 120), "grey": (120, 120, 120),
                "purple": (150, 50, 200), "orange": (255, 100, 50), "black": (40, 40, 40),
                "white": (200, 200, 200), "pink": (255, 100, 150), "gold": (210, 180, 50)
            }
            override_color = colors.get(color) or self._extract_color(prompt)
        else:
            override_color = self._extract_color(prompt)

        # 2. Compose Scene Components Independently
        self._generate_bg(sky, override_color)
        self._generate_ground(ground, override_color)
        self._generate_pipe(pipe, override_color)
        
        print("Assets Composed & Saved.")

    # --- Component Generators ---
    
    def _generate_bg(self, style, color):
        """ Generates bg.png based on Sky Style """
        if style == 'space':
            self._bg_gradient((0,0,20), (10,10,60))
            self._add_stars(100)
            self._add_circle_planet((200, 200, 100), (400, 100), 40)
        elif style == 'hell':
            self._bg_gradient((60, 0, 0), (20, 0, 0))
            self._add_mountains((40, 0, 0))
        elif style == 'snow':
            self._bg_gradient((200, 240, 255), (255, 255, 255))
            self._add_snowflakes(30)
        elif style == 'desert':
            self._bg_gradient((255, 150, 50), (255, 220, 100))
            self._add_sun((255, 255, 0), (400, 80), 30)
        elif style == 'candy':
            self._bg_gradient((255, 200, 220), (255, 240, 250))
            self._add_clouds(5)
        elif style == 'matrix':
            self._simple_rect((512, 384), (0, 0, 0))
            for _ in range(50):
                x, y = random.randint(0, 512), random.randint(0, 384)
                h = random.randint(5, 20)
                pygame.draw.rect(self.surface, (0, 255, 0), (x, y, 2, h))
        elif style == 'industrial':
            self._bg_gradient((100, 100, 100), (150, 150, 140))
        elif style == 'night':
             self._bg_gradient((0, 0, 40), (20, 20, 80))
             self._add_stars(50)
        else: # Day/Nature
            self._bg_gradient((100, 200, 255), (200, 230, 255))
            self._add_clouds(5)
            
        if color: self._tint_last_save("bg.png", color)
        else: self._save("bg.png")

    def _generate_ground(self, style, color):
        """ Generates ground.png based on Ground Style """
        if style == 'ice':
            self._simple_rect((512, 96), (220, 240, 255))
        elif style == 'lava':
             self._simple_rect((512, 96), (80, 0, 0))
             self._add_lava_cracks()
        elif style == 'sand':
             self._simple_rect((512, 96), (230, 200, 100))
        elif style == 'rock':
             self._simple_rect((512, 96), (60, 60, 60))
        elif style == 'tech':
             self._simple_rect((512, 96), (0, 50, 0))
             pygame.draw.line(self.surface, (0,200,0), (0, 48), (512, 48), 2)
        elif style == 'chocolate':
             self._simple_rect((512, 96), (150, 100, 50))
        elif style == 'road':
             self._simple_rect((512, 96), (40, 40, 40))
             pygame.draw.line(self.surface, (255,255,255), (0, 48), (512, 48), 4)
        else: # Grass
            self._simple_rect((512, 96), (100, 200, 100))
            
        self._save("ground.png", size=(512, 96))
        
        # Optional: Tint ground only if specific override
        if color: self._tint_last_save("ground.png", color)

    def _generate_pipe(self, style, color):
        """ Generates pipe.png based on Pipe Style """
        base_col = color if color else (50, 200, 50)
        
        if style == 'icy':
            if not color: base_col = (150, 200, 255)
            self._classic_pipe(base_col)
        elif style == 'cactus':
            if not color: base_col = (50, 150, 50)
            self._rock_pipe(base_col) 
        elif style == 'rock' or style == 'rusty':
            if not color: base_col = (120, 80, 50) if style == 'rusty' else (80, 80, 80)
            self._rock_pipe(base_col)
        elif style == 'tech' or style == 'neon':
            if not color: base_col = (0, 255, 255)
            if style == 'neon': self._neon_pipe(base_col)
            else: self._tech_pipe(base_col)
        elif style == 'striped':
            # Reuse simplistic striped pipe logic
            self.surface = pygame.Surface((64, 384))
            self.surface.fill((255, 255, 255))
            for i in range(0, 384, 20):
                pygame.draw.rect(self.surface, (255, 100, 100), (0, i, 64, 10))
            if color: 
                img = self.surface.copy()
                tin = self._apply_tint(img, color)
                self.surface.blit(tin, (0,0))
            self._save("pipe.png", size=(64,384))
            return 
        else: # Green/Classic
            self._classic_pipe(base_col)
            
        self._save("pipe.png", size=(64,384))



    def _extract_color(self, prompt):
        colors = {
            "red": (200, 50, 50), "green": (50, 200, 50), "blue": (50, 100, 255),
            "yellow": (200, 200, 50), "gray": (120, 120, 120), "grey": (120, 120, 120),
            "purple": (150, 50, 200), "orange": (255, 100, 50), "black": (40, 40, 40),
            "white": (200, 200, 200), "pink": (255, 100, 150), "gold": (210, 180, 50)
        }
        for name, rgb in colors.items():
            if name in prompt:
                return rgb
        return None

    def _load_and_tint_theme(self, theme_name, color):
        """ Loads pngs from assets/layout/{theme} and applies tint if color is set """
        theme_path = os.path.join(self.asset_lib_path, theme_name)
        
        files = ["bg.png", "ground.png", "pipe.png"]
        success = True
        for f in files:
            try:
                src = os.path.join(theme_path, f)
                if not os.path.exists(src):
                    print(f"Missing asset: {src}")
                    success = False
                    continue
                    
                img = pygame.image.load(src).convert_alpha()
                
                # Apply Tint if color provided
                if color:
                    img = self._apply_tint(img, color)
                
                # Save to session
                pygame.image.save(img, os.path.join(self.save_dir, f))
            except Exception as e:
                print(f"Error loading asset {f} for theme {theme_name}: {e}")
                success = False
        return success
    
    def _apply_tint(self, surface, color):
        """ Multiplies the surface by the color """
        tint_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        tint_surf.fill((color[0], color[1], color[2], 255))
        working_copy = surface.copy()
        # BLEND_MULT is cleaner for tinting
        working_copy.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        return working_copy

    # --- Drawing Primitives ---
    def _bg_gradient(self, c1, c2):
        self.surface = pygame.Surface((512, 384))
        h = 384
        for y in range(h):
            r = c1[0] + (c2[0] - c1[0]) * y / h
            g = c1[1] + (c2[1] - c1[1]) * y / h
            b = c1[2] + (c2[2] - c1[2]) * y / h
            pygame.draw.line(self.surface, (r,g,b), (0,y), (512,y))

    def _simple_rect(self, size, color):
        self.surface = pygame.Surface(size)
        self.surface.fill(color)

    def _add_stars(self, count):
        for _ in range(count):
            x, y = random.randint(0, 512), random.randint(0, 384)
            pygame.draw.circle(self.surface, (255, 255, 255), (x, y), 1 if random.random() > 0.1 else 2)

    def _add_circle_planet(self, color, pos, radius):
        pygame.draw.circle(self.surface, color, pos, radius)

    def _add_clouds(self, count):
        for _ in range(count):
            x, y = random.randint(0, 512), random.randint(50, 200)
            pygame.draw.circle(self.surface, (255, 255, 255), (x, y), 20)
            pygame.draw.circle(self.surface, (255, 255, 255), (x+15, y+5), 25)
            pygame.draw.circle(self.surface, (255, 255, 255), (x-15, y+5), 25)

    def _add_mountains(self, color):
        points = [(0, 384)]
        x = 0
        while x < 512:
            w = random.randint(50, 150)
            h = random.randint(100, 250)
            points.append((x + w//2, 384 - h))
            points.append((x + w, 384))
            x += w
        points.append((512, 384))
        pygame.draw.polygon(self.surface, color, points)

    def _add_buildings(self, color):
        count = 10
        w = 512 // count
        for i in range(count):
            h = random.randint(50, 200)
            pygame.draw.rect(self.surface, color, (i*w + 5, 384-h, w-10, h))
            # Windows
            for wy in range(384-h+10, 384, 15):
                for wx in range(i*w + 10, i*w + w - 10, 10):
                    if random.random() > 0.3:
                         pygame.draw.rect(self.surface, (255, 255, 0), (wx, wy, 4, 8))

    def _add_lava_cracks(self):
        for _ in range(10):
            x = random.randint(0, 512)
            pygame.draw.line(self.surface, (255, 100, 0), (x, 0), (x+random.randint(-20,20), 96), 2)

    def _add_craters(self, count):
         for _ in range(count):
            x, y = random.randint(0, 512), random.randint(0, 96)
            pygame.draw.circle(self.surface, (80, 80, 80), (x, y), random.randint(5, 15))

    def _add_bubbles(self, count):
         for _ in range(count):
            pygame.draw.circle(self.surface, (255, 255, 255), (random.randint(0, 512), random.randint(0, 384)), random.randint(1, 4), 1)

    def _tech_pipe(self, color):
        self._classic_pipe(color)
        pygame.draw.line(self.surface, (200, 200, 255), (10, 0), (10, 384), 2)
        pygame.draw.line(self.surface, (200, 200, 255), (50, 0), (50, 384), 2)

    def _rock_pipe(self, color):
        self._classic_pipe(color)
        # Add texture
        for _ in range(20):
             pygame.draw.circle(self.surface, (40, 20, 20), (random.randint(0, 64), random.randint(0, 384)), 5)

    def _neon_pipe(self, color):
        self.surface = pygame.Surface((64, 384), pygame.SRCALPHA)
        self.surface.fill((20, 20, 20))
        pygame.draw.rect(self.surface, color, (0,0,64,384), 2)
        pygame.draw.rect(self.surface, color, (15,0,34,384), 1)

    def _classic_pipe(self, color):
        self.surface = pygame.Surface((64, 384), pygame.SRCALPHA)
        self.surface.fill(color)
        pygame.draw.rect(self.surface, (0,0,0), (0,0,64,384), 4) # Outline
        pygame.draw.rect(self.surface, [min(c+30, 255) for c in color], (4, 4, 10, 376)) # Highlight

    def _save(self, filename, size=None):
        if size and self.surface.get_size() != size:
             self.surface = pygame.transform.scale(self.surface, size)
        pygame.image.save(self.surface, os.path.join(self.save_dir, filename))



# ==========================================
# 3. MANAGER
# ==========================================
class CoCreativeManager:
    def __init__(self, theme_manager):
        self.theme_manager = theme_manager
        self.session_assets_dir = os.path.join("assets", "session")
        if not os.path.exists(self.session_assets_dir):
            os.makedirs(self.session_assets_dir)
        
        # Use Real Semantic Brain
        self.ai_brain = SemanticBrain()
        self.ai_art = GenerativeArt(self.session_assets_dir)

    def process_prompt(self, text, current_params):
        if not text:
            return current_params
        
        print(f"--- Co-Creating with Local LLM: '{text}' ---")
        
        # 1. Analyze with LLM
        new_params = self.ai_brain.analyze(text, current_params)
        
        # 2. Extract Tags (Granular)
        llm_color = new_params.pop('llm_color', None)
        sky = new_params.pop('sky_style', 'day')
        ground = new_params.pop('ground_style', 'grass')
        pipe = new_params.pop('pipe_style', 'green')
        
        # 3. Generate Art (Compositional)
        self.ai_art.create_theme(text, sky, ground, pipe, llm_color)
        
        # 4. Apply Art
        self._apply_theme()
        
        return new_params

    def generate_proposal(self, score, time_alive):
        """ Asks the AI to propose a change based on performance """
        return self.ai_brain.propose_change(score, time_alive)

    def _apply_theme(self):
        try:
            self.theme_manager.themes["custom"] = {
                "background": pygame.image.load(os.path.join(self.session_assets_dir, "bg.png")).convert(),
                "pipe": pygame.image.load(os.path.join(self.session_assets_dir, "pipe.png")).convert_alpha(),
                "ground": pygame.image.load(os.path.join(self.session_assets_dir, "ground.png")).convert()
            }
            self.theme_manager.set_theme("custom")
        except Exception as e:
            print(f"Error applying theme: {e}")
