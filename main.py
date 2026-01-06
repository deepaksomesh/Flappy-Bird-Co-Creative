import pygame
import sys
from settings import WIDTH, HEIGHT, GROUND_HEIGHT
from world import World
from theme import ThemeManager
from sound import play as soundPlay
from game import GameIndicator
from ui_components import InputBox
# Layer 2 & 3
from cocreative import CoCreativeManager
from creative_state import CreativeState

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
pygame.display.set_caption("Super Flappy Bird Co-Creative")
theme = ThemeManager()

# ==========================================
# STATE MACHINE CONSTANTS
# ==========================================
STATE_WAITING = "WAITING"         # "Press SPACE"
STATE_PLAYING = "PLAYING"         # Physics Active
STATE_GAME_OVER = "GAME_OVER"     # Bird Dead
STATE_NEGOTIATING = "NEGOTIATING" # AI Suggestion Overlay
STATE_INPUT = "INPUT"             # User typing
STATE_FEEDBACK = "FEEDBACK"       # User rating level
STATE_COUNTDOWN = "COUNTDOWN"     # 3..2..1


class Main:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 40)
        
        # Core Components
        self.world = World(screen, theme)
        self.creative_state = CreativeState()
        self.cocreative = CoCreativeManager(theme)
        
        # UI Components
        self.input_box = InputBox(WIDTH // 2 - 150, HEIGHT // 2 - 16, 300, 32)
        
        # State
        self.state = STATE_WAITING
        self.ai_proposal = None
        self.proposal_handled = False # Ensures we only negotiate once per death
        
        # Timers
        self.countdown_val = 3
        self.countdown_timer = 0

    def main_loop(self):
        while True:
            dt = self.clock.tick(60) / 1000.0 # Delta time if needed
            self.handle_events()
            self.update()
            self.draw()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # --- GLOBAL INPUTS ---
            if event.type == pygame.KEYDOWN:
                # Open Input Box (SHIFT) - Valid in most states except NEGOTIATING/INPUT/FEEDBACK
                if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT) and self.state not in (STATE_INPUT, STATE_NEGOTIATING, STATE_FEEDBACK):
                    self._enter_input_mode()
                    continue

            # --- STATE SPECIFIC ---
            if self.state == STATE_INPUT:
                self._handle_input(event)
            
            elif self.state == STATE_NEGOTIATING:
                self._handle_negotiation(event)
            
            elif self.state == STATE_FEEDBACK:
                self._handle_feedback(event)
                
            elif self.state == STATE_WAITING:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    soundPlay("jump")
                    self.world.update("jump")
                    self.state = STATE_PLAYING

            elif self.state == STATE_PLAYING:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    soundPlay("jump")
                    self.world.update("jump")

            elif self.state == STATE_GAME_OVER:
                # Press Space to simple restart if no negotiation
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                     self._restart_game_round()

    def update(self):
        # Physics only update in PLAYING
        # World.update() handles physics, collision, etc.
        
        if self.state == STATE_PLAYING:
            self.world.update()
            if self.world.game_over:
                self._on_death()

        elif self.state == STATE_WAITING:
             pass
        
        elif self.state == STATE_COUNTDOWN:
            now = pygame.time.get_ticks()
            if now - self.countdown_timer > 1000:
                self.countdown_val -= 1
                self.countdown_timer = now
                if self.countdown_val <= 0:
                    self.state = STATE_PLAYING
                    self.world.update("jump")

        # Input box update
        if self.state == STATE_INPUT:
            self.input_box.update()
            
        # Ground scroll (visuals)
        self.world._scroll_world() 

    def draw(self):
        # Draw World
        self.screen.fill((0,0,0)) # Clear
        
        bg = theme.get('background')
        if bg: self.screen.blit(pygame.transform.scale(bg, (WIDTH, HEIGHT)), (0,0))
        
        self.world.draw()
        
        # Draw Ground
        ground = theme.get('ground')
        if ground: self.screen.blit(ground, (self.world.world_shift, HEIGHT))
        
        # --- OVERLAYS ---
        
        if self.state == STATE_WAITING:
             self._draw_text("Press SPACE to Start the Game", 50, WIDTH//2, HEIGHT//2, (255, 255, 255))
             
        if self.state == STATE_PLAYING:
             if self.world.last_score > 10:
                  self._draw_text("You are playing well, try pressing SHIFT and Have fun!!!", 30, WIDTH//2, 100, (255, 255, 0))

        if self.state == STATE_GAME_OVER and not self.ai_proposal:
             self.world.game.instructions()

        if self.state == STATE_NEGOTIATING:
             self._draw_overlay((0,0,20, 200)) # Blue Tint
             self._draw_negotiation_ui()
             
        if self.state == STATE_FEEDBACK:
             self._draw_overlay((0,0,0, 220)) 
             self._draw_feedback_ui()

        if self.state == STATE_INPUT:
             self._draw_overlay((0,0,0, 150))
             self.input_box.draw(self.screen)

        if self.state == STATE_COUNTDOWN:
             self._draw_text(str(self.countdown_val), 80, WIDTH//2, HEIGHT//2, (255,255,255))
        
        pygame.display.update()

    # ==========================
    # LOGIC HELPERS
    # ==========================

    def _on_death(self):
        self.proposal_handled = False
        
        # If in Creative Mode, we MUST get feedback first
        if self.creative_state.is_active():
            self.state = STATE_FEEDBACK
        else:
            self.state = STATE_GAME_OVER
            self._check_for_ai_intervention()

    def _restart_game_round(self):
        """ Resets entities. Preserves Creative State params. """
        self.world.reset_game_logic() 
        self.world.set_physics_params(self.creative_state.current_params) 
        self.state = STATE_WAITING

    def _check_for_ai_intervention(self):
        if self.proposal_handled: return
        self.proposal_handled = True
        
        snapshot = self.world.get_snapshot()
        score = snapshot['score']
        
        if score >= 0:
             # PASS FEEDBACK HISTORY TO AI!
             good_examples = self.creative_state.get_good_examples()
             self.ai_proposal = self.cocreative.generate_proposal(score, 0, snapshot, good_examples)
             self.state = STATE_NEGOTIATING
             print(f"Negotiating: {self.ai_proposal}")

    def _apply_new_level(self, prompt, params):
        print("Committing new level...")
        self.creative_state.activate(params, prompt) # Track prompt
        self.world.reset_game_logic()
        self.world.set_physics_params(params)
        self.state = STATE_COUNTDOWN
        self.countdown_val = 3
        self.countdown_timer = pygame.time.get_ticks()
    
    def _handle_feedback(self, event):
        if event.type == pygame.KEYDOWN:
            rating = None
            if event.key == pygame.K_1: rating = "good"
            elif event.key == pygame.K_2: rating = "okay"
            elif event.key == pygame.K_3: rating = "bad"
            
            if rating:
                self.creative_state.register_feedback(rating)
                # After feedback, we proceed to AI Proposal loop or just Game Over
                # User says: "Appears as soon as player crashes... AI should save... and [then?]"
                # Usually we want to negotiate NEXT change.
                self.state = STATE_GAME_OVER
                self._check_for_ai_intervention()

    def _draw_feedback_ui(self):
        self._draw_text("HOW WAS THIS LEVEL?", 50, WIDTH//2, HEIGHT//2 - 80, (255, 255, 0))
        self._draw_text(f"Current: {self.creative_state.current_prompt}", 30, WIDTH//2, HEIGHT//2 - 30, (200, 200, 200))
        
        self._draw_text("[1] GOOD (Keep it)", 40, WIDTH//2, HEIGHT//2 + 40, (0, 255, 0))
        self._draw_text("[2] OKAY (Improve it)", 40, WIDTH//2, HEIGHT//2 + 90, (255, 200, 0))
        self._draw_text("[3] BAD (Reject it)", 40, WIDTH//2, HEIGHT//2 + 140, (255, 50, 50))
    
    # ... (Rest of Input/Negotiation handlers same)

    def _handle_input(self, event):
        prompt = self.input_box.handle_event(event)
        if self.input_box.done:
            if prompt:
                if prompt.strip().lower() == "reset":
                    self.creative_state.reset()
                    self.world.reset_game_logic()
                    self.world.reset_physics_to_default() # Reset internal defaults
                    self.state = STATE_WAITING
                else:
                    # Process
                    current = self.creative_state.get_params()
                    new_params = self.cocreative.process_prompt(prompt, current)
                    self._apply_new_level(prompt, new_params)
            else:
                 # Cancelled
                 self.state = STATE_WAITING
                 
            self.input_box.active = False

    def _enter_input_mode(self, text=""):
        self.state = STATE_INPUT
        self.input_box.reset()
        if text: self.input_box.set_text(text)
        self.input_box.active = True

    def _handle_negotiation(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_y:
                # Accept
                current = self.creative_state.get_params()
                new_params = self.cocreative.process_prompt(self.ai_proposal, current)
                self._apply_new_level(self.ai_proposal, new_params)
                self.ai_proposal = None
                
            elif event.key == pygame.K_n:
                # Reject -> Provide Feedback? 
                # Req: "User choice Reject -> continue normally"
                # Req: "Feedback Collected... Only after AI-generated levels"
                # If we reject, we didn't play it. So no feedback yet?
                self.state = STATE_GAME_OVER
                self.ai_proposal = None
                self.world.game.instructions() # Show score etc
                
            elif event.key == pygame.K_m:
                # Modify
                self._enter_input_mode("")
                self.ai_proposal = None

    def _draw_negotiation_ui(self):
        self._draw_text("AI SUGGESTION", 50, WIDTH//2, HEIGHT//2 - 80, (0, 255, 255))
        self._draw_text(f"\"{self.ai_proposal}\"", 30, WIDTH//2, HEIGHT//2, (255, 255, 255))
        self._draw_text("[Y] Accept  [N] Reject  [M] Modify", 30, WIDTH//2, HEIGHT//2 + 60, (200, 200, 200))

    def _draw_overlay(self, color):
        s = pygame.Surface((WIDTH, HEIGHT+GROUND_HEIGHT))
        s.set_alpha(color[3])
        s.fill(color[:3])
        self.screen.blit(s, (0,0))

    def _draw_text(self, text, size, x, y, color):
        font = pygame.font.Font(None, size)
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(x,y))
        self.screen.blit(surf, rect)


if __name__ == "__main__":
    game = Main(screen)
    soundPlay("background")
    game.main_loop()
