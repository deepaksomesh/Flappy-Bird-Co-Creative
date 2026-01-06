
import pygame
from pipe import Pipe
from bird import Bird
from game import GameIndicator
from settings import WIDTH, HEIGHT, PIPE_SIZE, PIPE_GAP, DEFAULT_PIPE_PAIR, PIPE_PATTERNS, THEMES
import random
from sound import play, stop
import numpy as np

class World:
    def __init__(self, screen, theme, isMulti=False):
        self.screen = screen
        self.theme = theme
        self.multi_mode = isMulti
        
        # Physics Parameters (The Truth)
        self.speed = -6.0
        self.gravity = 0.5
        self.vel = 10.0
        self.dist = 4
        self.gap_multiplier = 1.0
        self.pipe_move_speed = 0.0
        
        # Game State
        self.playing = False
        self.game_over = False
        self.game_over_sound = False
        self.last_score = 0
        self.last_milestone = 0
        
        # Entities
        self.upcoming_pipes = pygame.sprite.Group()
        self.player = pygame.sprite.GroupSingle()
        self.game = GameIndicator(screen, theme)
        
        # Internal counters
        self.world_shift = 0
        self.pipe_count = 0
        self.pattern_index = 0
        self.current_pattern_name = 'zigzag'
        self.game_mode = "day"
        self.theme_switch_count = 15
        
        # RL / Analysis
        self.current_reward = 0.0
        self.scored_pipes = set()
        
        # Initialize
        self.reset_game_logic()

    def set_physics_params(self, params):
        """
        Layer 1 Interface: External forces set these parameters.
        World blindly obeys them.
        """
        print(f"World Physics Updated: {params}")
        if 'speed' in params: self.speed = params['speed']
        if 'gravity' in params: self.gravity = params['gravity']
        if 'vel' in params: self.vel = params['vel']
        if 'gap_multiplier' in params: self.gap_multiplier = params['gap_multiplier']
        if 'pipe_move_speed' in params: self.pipe_move_speed = params['pipe_move_speed']
        # 'dist' in params is optional in new parser but we keep it
        
        # Also update theme if present
        if 'game_mode' in params:
             self.game_mode = params['game_mode'] # Just a label fallback

    def get_params(self):
        return {
            "speed": self.speed,
            "gravity": self.gravity,
            "vel": self.vel,
            "gap_multiplier": self.gap_multiplier,
            "pipe_move_speed": self.pipe_move_speed
        }

    def reset_game_logic(self):
        """
        Resets ENTITIES and SCORE.
        Does NOT reset PHYSICS (Persistence Requirement).
        """
        self.upcoming_pipes.empty()
        self.player.empty()
        
        self.world_shift = 0
        self.pipe_count = 0
        self.pattern_index = 0
        self.last_score = 0
        self.scored_pipes.clear()
        
        # Bird Spawn
        bird = Bird((WIDTH // 4, HEIGHT // 2 - 50), 45)
        self.player.add(bird)
        
        # Pipe Spawn
        self._add_pipe()
        # Shift first pipe
        for pipe in self.upcoming_pipes:
            pipe.rect.x = WIDTH + 150
            
        self.playing = True
        self.game_over = False
        self.game_over_sound = False
    
    def reset_physics_to_default(self):
        """
        Explicit reset command.
        """
        self.speed = -6.0
        self.gravity = 0.5
        self.vel = 10.0
        self.gap_multiplier = 1.0
        self.pipe_move_speed = 0.0
        self.game_mode = "day"
        self.theme.set_theme("day")

    def update(self, player_event=None):
        # Input Handling
        if player_event == "jump" and self.player.sprite and not self.game_over:
            self.player.sprite.update(is_jump=True, game_mode=self.game_mode)
        
        # Physics Update
        self._apply_physics()
        self._scroll_world()
        self.upcoming_pipes.update(self.world_shift)
        
        # Generation
        rightmost = -float('inf')
        for pipe in self.upcoming_pipes:
            rightmost = max(rightmost, pipe.rect.right)
        if rightmost < WIDTH - (PIPE_SIZE * self.dist):
            self._add_pipe()
            
        # Collision
        self._handle_collision()
        self._update_score()

    def draw(self):
        self.upcoming_pipes.draw(self.screen)
        self.player.draw(self.screen)
        self.game.show_score(self.last_score)

    def _apply_physics(self):
        if self.player.sprite and not self.game_over:
            self.player.sprite.direction.y += self.gravity
            self.player.sprite.rect.y += self.player.sprite.direction.y

    def _scroll_world(self):
        # If auto-scaling is active (Non-Creative), logic would go here.
        # But per requirements, we want deterministic control from outside.
        # So we trust self.speed.
        self.world_shift = self.speed if self.playing and not self.game_over else 0

    def _add_pipe(self):
        # Logic to pick pattern
        if self.pipe_count % 5 == 0:
            self.current_pattern_name = random.choice(list(PIPE_PATTERNS.keys()))
            self.pattern_index = 0
        
        pattern = PIPE_PATTERNS[self.current_pattern_name]
        pipe_pairs = pattern.get("pairs")
        
        # Use stored multiplier
        current_gap = int(PIPE_GAP * self.gap_multiplier)
        
        if pipe_pairs:
             pair = pipe_pairs[self.pattern_index % len(pipe_pairs)]
             self.pattern_index += 1
        else:
             pair = random.choice(DEFAULT_PIPE_PAIR)
             
        top_h = pair[0] * PIPE_SIZE
        bottom_y = top_h + current_gap
        
        top = Pipe((WIDTH, top_h - HEIGHT), PIPE_SIZE, HEIGHT, True, self.game_mode, self.theme, self.pipe_move_speed)
        bottom = Pipe((WIDTH, bottom_y), PIPE_SIZE, HEIGHT, False, self.game_mode, self.theme, self.pipe_move_speed)
        
        self.upcoming_pipes.add(top)
        self.upcoming_pipes.add(bottom)
        self.pipe_count += 1

    def _handle_collision(self):
        if self.game_over: return
        bird = self.player.sprite
        if not bird: return
        
        hit = pygame.sprite.spritecollide(bird, self.upcoming_pipes, False, pygame.sprite.collide_mask)
        if hit or bird.rect.top <= 0 or bird.rect.bottom >= HEIGHT:
            self.game_over = True
            if not self.game_over_sound:
                play("hit")
                self.game_over_sound = True
            self.playing = False

    def _update_score(self):
        if self.game_over: return
        bird = self.player.sprite
        if not bird: return
        
        # Score Logic
        # Simplistic: check if passed pipes
        for pipe in self.upcoming_pipes:
            if not pipe.get_is_flipped() and pipe not in self.scored_pipes:
                if bird.rect.centerx > pipe.rect.centerx:
                    self.last_score += 1
                    self.scored_pipes.add(pipe)
                    play("score")

    def get_snapshot(self):
        """ Returns performance snapshot for AI """
        bird = self.player.sprite
        cause = "unknown"
        if bird:
            if bird.rect.bottom >= HEIGHT: cause = "ground"
            elif bird.rect.top <= 0: cause = "ceiling"
            else: cause = "pipe"
            
        return {
            "score": self.last_score,
            "cause": cause
        }