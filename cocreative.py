import os
import pygame
import json
import random
import re
import torch
import numpy as np

# semantic parser
class SemanticBrain:
    def __init__(self):
        # We process text locally using extensive synonym maps.
        # This acts as a "Semantic Engine" without the unpredictability of an LLM.
        self.pipe = None
        try:
            from transformers import pipeline
            # We keep LLM mostly for creative names/fallback, but not for core physics
            self.pipe = pipeline("text2text-generation", model="google/flan-t5-small", device="cpu") 
            print("Local LLM Loaded...")
        except:
            print("Local LLM Not Found. Switching to Deterministic Mode...")

        self.theme_synonyms = {
            'space': ['galaxy', 'cosmos', 'star', 'orbit', 'alien', 'planet'],
            'moon': ['lunar', 'crater', 'grey', 'dust'],
            'hell': ['lava', 'fire', 'inferno', 'demon', 'doom', 'evil', 'burn', 'red'],
            'snow': ['ice', 'cold', 'freeze', 'winter', 'chill', 'blizzard', 'white'],
            'desert': ['sand', 'dune', 'dry', 'hot', 'cactus', 'egypt'],
            'candy': ['sweet', 'sugar', 'pink', 'chocolate', 'cake', 'cookie'],
            'matrix': ['tech', 'cyber', 'code', 'hack', 'green', 'digital', 'neo'],
            'night': ['dark', 'evening', 'midnight', 'sleep'],
            'day': ['sun', 'bright', 'morning', 'light', 'sky'],
            'underwater': ['ocean', 'sea', 'fish', 'blue', 'water', 'dive', 'swim', 'bubble', 'deep'],
            'forest': ['tree', 'wood', 'jungle', 'nature', 'green', 'plant', 'leaf'],
            'sunset': ['orange', 'dusk', 'evening', 'purple', 'horizon']
        }
        
        # Regex patterns
        self.patterns = {}
        
        # Magnitude
        self.patterns['mag_high'] = re.compile(r"\b(val|very|super|insane|ultra|huge|extreme|large|max|lot)\b")
        self.patterns['mag_low'] = re.compile(r"\b(slightly|little|bit|tad|small|mild|kinda)\b")
        
        # Themes
        self.theme_patterns = {}
        for theme, synonyms in self.theme_synonyms.items():
            # This is to match theme name OR any synonym, with word boundaries
            regex_str = r"\b(" + theme + "|" + "|".join(synonyms) + r")\b"
            self.theme_patterns[theme] = re.compile(regex_str)
            
        # Movement of pipes
        self.patterns['move_start'] = re.compile(r"\b(moving|dynamic|oscillating|bobbing|shifting|variable|animated|alive)\b")
        self.patterns['move_stop'] = re.compile(r"\b(static|stop moving|stable|fixed|freeze)\b|no motion")
        
        # Speed
        self.patterns['fast'] = re.compile(r"\b(faster|fast|quick|dash|race|zoom|rapid|sprint|hurry|swift|speed\s*up|high\s*speed)\b")
        self.patterns['slow'] = re.compile(r"\b(slower|slow|crawl|creep|lethargic|slug|chill|relax|easy|lofi|delay|slow\s*down)\b")
        
        # Gravity
        self.patterns['heavy'] = re.compile(r"\b(heavy|heavier|lead|iron|rock|stone|drop|fall|strong|crush|intense)\b|\b(increase|more|high)\s*grav|\bless\s*float")
        self.patterns['light'] = re.compile(r"\b(light|lighter|float|feather|balloon|moon|space|drift|soft|glide|fly)\b|\b(decrease|less|low)\s*grav|\bunderwater\b")
        
        # Gap size of pipes
        self.patterns['gap_wide'] = re.compile(r"\b(wider|wide|broad|expand|huge|open|large|spacious)\b|\b(increase|big|more)\s*gap")
        self.patterns['gap_narrow'] = re.compile(r"\b(narrow|tiny|small|tight|squeeze|hard|close|shut)\b|\b(reduce|decrease|less|small)\s*gap")

    def _classify_theme_llm(self, text, candidates):
        """ we ask LLM to pick the best theme from the list """
        if not self.pipe: return None
        try:
            options = ", ".join(candidates)
            prompt = f"Concept: {text}. Best Match from [{options}]? Answer:"
            out = self.pipe(prompt, max_new_tokens=5)[0]['generated_text'].lower()
            
            for c in candidates:
                if c in out: return c
        except:
            pass
        return None

    def _classify_movement_llm(self, text):
        """ we ask LLM if text implies movement """
        if not self.pipe: return False
        try:
            prompt = f"Does the phrase '{text}' imply moving obstacles or dynamic hazards? Answer Yes or No."
            out = self.pipe(prompt, max_new_tokens=3)[0]['generated_text'].lower()
            return "yes" in out
        except:
            pass
        return False

    def analyze(self, prompt, current_params, feedback_history=[]):
        """
        We use a 'Bag of Words' + 'Modifier' approach to semantically understand request.
        """
        print(f"Analyzing Prompt: '{prompt}'")
        p = current_params.copy()
        
        p['pipe_move_speed'] = 0.0 
        
        text = prompt.lower()
        
        # How intense is the request? (magnitude)
        magnitude = 1.0
        
        if self.patterns['mag_high'].search(text):
            magnitude = 2.0
        elif self.patterns['mag_low'].search(text):
            magnitude = 0.5
            
        print(f"Detected Semantic Magnitude: {magnitude}x")

        # Theme & Presets
        styles = {
            'space':   {'sky': 'space', 'ground': 'tech',  'pipe': 'tech',    'grav': 0.25, 'speed': -6.0, 'pipe_move_speed': 0.0},
            'moon':    {'sky': 'space', 'ground': 'rock',  'pipe': 'rock',    'grav': 0.15, 'speed': -5.0, 'pipe_move_speed': 0.0},
            'hell':    {'sky': 'hell',  'ground': 'lava',  'pipe': 'rock',    'grav': 0.8,  'speed': -9.0, 'pipe_move_speed': 0.0},
            'snow':    {'sky': 'snow',  'ground': 'ice',   'pipe': 'icy',     'grav': 0.5,  'speed': -5.0, 'pipe_move_speed': 0.0},
            'ice':     {'sky': 'snow',  'ground': 'ice',   'pipe': 'icy',     'grav': 0.5,  'speed': -5.0, 'pipe_move_speed': 0.0},
            'desert':  {'sky': 'desert','ground': 'sand',  'pipe': 'cactus',  'grav': 0.6,  'speed': -7.0, 'pipe_move_speed': 0.0},
            'candy':   {'sky': 'candy', 'ground': 'chocolate','pipe': 'striped', 'grav': 0.4, 'speed': -6.0, 'pipe_move_speed': 0.0},
            'matrix':  {'sky': 'matrix','ground': 'tech',  'pipe': 'neon',    'grav': 0.5,  'speed': -10.0, 'pipe_move_speed': 0.0},
            'night':   {'sky': 'night', 'ground': 'grass', 'pipe': 'green',   'grav': 0.5,  'speed': -6.0, 'pipe_move_speed': 0.0},
            'day':     {'sky': 'day',   'ground': 'grass', 'pipe': 'green',   'grav': 0.5,  'speed': -6.0, 'pipe_move_speed': 0.0},
            'underwater': {'sky': 'underwater', 'ground': 'sand', 'pipe': 'green', 'grav': 0.3, 'speed': -5.0, 'pipe_move_speed': 0.0},
            'forest':     {'sky': 'forest',     'ground': 'grass','pipe': 'wood',  'grav': 0.5, 'speed': -6.0, 'pipe_move_speed': 0.0},
            'sunset':     {'sky': 'sunset',     'ground': 'sand', 'pipe': 'rust',  'grav': 0.5, 'speed': -7.0, 'pipe_move_speed': 0.0}
        }
        
        found_style = None
        
        
        # Regex Theme Match
        for theme_name, regex in self.theme_patterns.items():
            if regex.search(text):
                found_style = theme_name
                break
            
        # LLM Abstract Match
        if not found_style and self.pipe:
            if len(text.split()) > 2: 
                print("Attempting LLM Theme Classification...")
                matches = list(styles.keys())
                llm_guess = self._classify_theme_llm(text, matches)
                if llm_guess:
                   print(f"LLM Predicted Theme: {llm_guess.upper()}")
                   found_style = llm_guess

        if found_style:
            print(f"Applied Theme Preset: {found_style.upper()}")
            preset = styles[found_style]
            p['sky_style'] = preset.get('sky', p.get('sky_style', 'day'))
            p['ground_style'] = preset.get('ground', p.get('ground_style', 'grass'))
            p['pipe_style'] = preset.get('pipe', p.get('pipe_style', 'green'))
            p['gravity'] = preset.get('grav', p['gravity'])
            p['speed'] = preset.get('speed', p['speed'])
            p['pipe_move_speed'] = preset.get('pipe_move_speed', 0.0)

        # Physics Intent Parsing (Regex)
    
        is_moving = False
        if self.patterns['move_start'].search(text):
             is_moving = True
        elif self.pipe and not self.patterns['move_stop'].search(text):
             if len(text.split()) > 3:
                 if self._classify_movement_llm(text):
                     is_moving = True

        if is_moving:
            p['pipe_move_speed'] = 1.0 * magnitude
            print(f"Logic: MOVING PIPES ({p['pipe_move_speed']})")
        elif self.patterns['move_stop'].search(text):
            p['pipe_move_speed'] = 0.0
            print(f"Logic: STATIC PIPES ({p['pipe_move_speed']})")
        
        # Speed
        delta_speed = 3.0 * magnitude
        if self.patterns['fast'].search(text):
            p['speed'] = max(p['speed'] - delta_speed, -15.0)
            print(f" > Logic: FASTER ({delta_speed})")
        elif self.patterns['slow'].search(text):
            p['speed'] = min(p['speed'] + delta_speed, -2.0)
            print(f" > Logic: SLOWER ({delta_speed})")

        # Gravity
        delta_grav = 0.2 * magnitude
        if self.patterns['heavy'].search(text):
            p['gravity'] = min(p['gravity'] + delta_grav, 1.3)
            print(f" > Logic: HEAVIER ({delta_grav})")
        elif self.patterns['light'].search(text):
            p['gravity'] = max(p['gravity'] - delta_grav, 0.1)
            print(f" > Logic: LIGHTER ({delta_grav})")

        # Gap
        delta_gap = 0.3 * magnitude
        if self.patterns['gap_wide'].search(text):
            p['gap_multiplier'] = min(p['gap_multiplier'] + delta_gap, 2.5)
            print(f" > Logic: WIDER GAP ({delta_gap})")
        elif self.patterns['gap_narrow'].search(text):
            p['gap_multiplier'] = max(p['gap_multiplier'] - delta_gap, 0.5)
            print(f" > Logic: NARROWER GAP ({delta_gap})")

        if p['gravity'] < 0.25: p['vel'] = 7.0 
        elif p['gravity'] < 0.4: p['vel'] = 8.0
        elif p['gravity'] > 0.9: p['vel'] = 14.0
        elif p['gravity'] > 0.7: p['vel'] = 12.0
        else: p['vel'] = 10.0

        print(f"Final Params: Spd:{p['speed']:.1f} Grv:{p['gravity']:.1f} Gap:{p['gap_multiplier']:.1f}")
        return p


    def propose_change(self, score, time_alive, performance_snapshot=None, good_examples=None):
        """ The AI suggests a new level setting """
        if not self.pipe: return "Make it easier"
        
        # Mad-Libs Style Generator 
        adjectives = ["intense", "chill", "crazy", "dark", "neon", "retro", "heavy", "floaty", "fast", "weird"]
        nouns = ["mode", "world", "zone", "dimension", "level", "vibes"]
        actions = ["with wider gaps", "with high gravity", "but faster", "but slower", "with tiny gaps", "style", "with moving pipes"]
        themes = ["space", "hell", "snow", "matrix", "desert", "candy", "night", "underwater", "forest", "sunset"]
        
        t = random.choice(themes)
        adj = random.choice(adjectives)
        noun = random.choice(nouns)
        action = random.choice(actions)
        fallback = f"{adj.capitalize()} {t} {noun} {action}"
        
        # Score Logic
        if score < 2:
            return f"Easier {random.choice(['snow', 'candy'])} mode with wide gaps"
        if score > 10:
            return f"Impossible {random.choice(['hell', 'matrix'])} mode with speed"

        # USE GOOD EXAMPLES for In-Context Learning
        if good_examples:
            # 50% chance to mutate a good example
            if random.random() < 0.5:
                base = random.choice(good_examples).replace('"','').replace("'","")
                return f"{base} but {random.choice(['faster', 'harder', 'weird'])}"

        # Prompting LLM just for a cool name/idea
        try:
            prompt = f"Invent a creative game mode name like 'Neon Space Hover'. Status: Score {score}."
            out = self.pipe(prompt, max_new_tokens=10, do_sample=True, temperature=0.9)[0]['generated_text']
            if len(out) > 5: return out
        except:
            pass
            
        return fallback


class GenerativeArt:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.surface = None 
        self.asset_lib_path = os.path.join("assets", "layout")

    def create_theme(self, prompt, sky='day', ground='grass', pipe='green', color=None):
        self._generate_bg(sky, color)
        self._generate_ground(ground, color)
        self._generate_pipe(pipe, color)

    def _generate_bg(self, style, color):
        if style == 'space': self._draw_gradient((0,0,20), (10,10,60)); self._add_stars(100)
        elif style == 'hell': self._draw_gradient((60,0,0), (20,0,0)); self._add_mountains((40,0,0))
        elif style == 'snow': self._draw_gradient((200,240,255), (255,255,255)); self._add_snowbubbles(30)
        elif style == 'desert': self._draw_gradient((255,150,50), (255,220,100)); 
        elif style == 'night': self._draw_gradient((0,0,40), (20,20,80)); self._add_stars(50)
        elif style == 'matrix': self._simple_fill((0,0,0)); self._add_matrix_rain()
        elif style == 'underwater': self._draw_gradient((0,50,150), (0,150,200)); self._add_bubbles(40)
        elif style == 'forest': self._draw_gradient((50,150,50), (150,220,150)); self._add_trees(10)
        elif style == 'sunset': self._draw_gradient((100,50,100), (255,150,50)); self._add_sun()
        
        else: self._draw_gradient((100,200,255), (200,230,255)); self._add_clouds(5) # Day
        
        self._save("bg.png", color)

    def _generate_ground(self, style, color):
        if style == 'ice': self._simple_fill((220,240,255), (512,96))
        elif style == 'lava': self._simple_fill((80,0,0), (512,96))
        elif style == 'tech': self._simple_fill((0,50,0), (512,96)); self._draw_line((0,48),(512,48))
        elif style == 'sand': self._simple_fill((240,230,140), (512,96)) # Sand
        else: self._simple_fill((100,200,100), (512,96)) # Grass
        self._save("ground.png", color, (512,96))

    def _generate_pipe(self, style, color):
        # Base colors
        c = (50,200,50)
        if style == 'icy': c = (150,200,255)
        elif style == 'rock': c = (80,80,80)
        elif style == 'rust': c = (120,80,50)
        elif style == 'neon': c = (0,255,255)
        elif style == 'wood': c = (100,60,30)
        
        self.surface = pygame.Surface((64,384))
        self.surface.fill(c)
        pygame.draw.rect(self.surface, (0,0,0), (0,0,64,384), 4)
        
        if style == 'neon': pygame.draw.rect(self.surface, (255,255,255), (10,0,10,384))
        if style == 'wood': pygame.draw.rect(self.surface, (90,50,20), (15,0,5,384)); pygame.draw.rect(self.surface, (90,50,20), (45,0,5,384)) 
        
        self._save("pipe.png", color, (64,384))

    def _draw_gradient(self, c1, c2):
        self.surface = pygame.Surface((512,384))
        for y in range(384):
             t = y/384
             r = c1[0] + (c2[0]-c1[0])*t
             g = c1[1] + (c2[1]-c1[1])*t
             b = c1[2] + (c2[2]-c1[2])*t
             pygame.draw.line(self.surface, (int(r),int(g),int(b)), (0,y), (512,y))
    
    def _simple_fill(self, color, size=(512,384)):
        self.surface = pygame.Surface(size)
        self.surface.fill(color)
        
    def _add_stars(self, n):
        for _ in range(n):
            pygame.draw.circle(self.surface, (255,255,255), (random.randint(0,512), random.randint(0,384)), 1)
            
    def _add_bubbles(self, n):
        for _ in range(n):
             pygame.draw.circle(self.surface, (255,255,255), (random.randint(0,512), random.randint(0,384)), random.randint(2,6), 1)

    def _add_trees(self, n):
        for i in range(n):
             x = random.randint(0,512)
             pygame.draw.polygon(self.surface, (30,100,30), [(x,384), (x+30,384), (x+15,384-random.randint(50,150))])

    def _add_sun(self):
         pygame.draw.circle(self.surface, (255,200,50), (256, 384), 100)

    def _add_clouds(self, n):
        for _ in range(n):
             pygame.draw.circle(self.surface, (255,255,255), (random.randint(0,512), random.randint(50,200)), 20)

    def _add_snowbubbles(self, n):
        self._add_stars(n)
        
    def _add_mountains(self, color):
         pygame.draw.polygon(self.surface, color, [(0,384), (100,250), (200,384), (300,200), (512,384)])

    def _add_matrix_rain(self):
         for _ in range(50):
              x,h = random.randint(0,512), random.randint(5,30)
              pygame.draw.rect(self.surface, (0,255,0), (x, random.randint(0,384), 2, h))

    def _draw_line(self, p1, p2):
         pygame.draw.line(self.surface, (200,255,200), p1, p2, 2)
    def _save(self, name, tint=None, size=None):
        if size and self.surface.get_size() != size: self.surface = pygame.transform.scale(self.surface, size)
        pygame.image.save(self.surface, os.path.join(self.save_dir, name))


class CoCreativeManager:
    def __init__(self, theme_manager):
        self.theme_manager = theme_manager
        self.session_assets_dir = os.path.join("assets", "session")
        if not os.path.exists(self.session_assets_dir): os.makedirs(self.session_assets_dir)
        self.ai_brain = SemanticBrain()
        self.ai_art = GenerativeArt(self.session_assets_dir)

    def process_prompt(self, text, current_params):
        if not text: return current_params
        print(f"\nCo-Creating: '{text}'")
        new_params = self.ai_brain.analyze(text, current_params)
        sky = new_params.get('sky_style', 'day')
        ground = new_params.get('ground_style', 'grass')
        pipe = new_params.get('pipe_style', 'green')
        self.ai_art.create_theme(text, sky, ground, pipe, None)
        self._apply_theme()
        return new_params

    def generate_proposal(self, score, time_alive, performance_snapshot=None, good_examples=None):
        return self.ai_brain.propose_change(score, time_alive, performance_snapshot, good_examples)

    def _apply_theme(self):
        try:
            self.theme_manager.themes["custom"] = {
                "background": pygame.image.load(os.path.join(self.session_assets_dir, "bg.png")).convert(),
                "pipe": pygame.image.load(os.path.join(self.session_assets_dir, "pipe.png")).convert_alpha(),
                "ground": pygame.image.load(os.path.join(self.session_assets_dir, "ground.png")).convert()
            }
            self.theme_manager.set_theme("custom")
        except Exception as e: print(f"Error: {e}")
