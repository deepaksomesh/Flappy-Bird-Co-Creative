import pygame
import sys

from settings import WIDTH, HEIGHT, GROUND_HEIGHT
from world import World
from theme import ThemeManager
from sound import play as soundPlay
from ui_components import InputBox

from cocreative import CoCreativeManager
from creative_state import CreativeState

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
pygame.display.set_caption("Super Flappy Bird Co-Creative")
theme = ThemeManager()

STATE_WAITING = "WAITING"        
STATE_PLAYING = "PLAYING"         
STATE_GAME_OVER = "GAME_OVER"     
STATE_NEGOTIATING = "NEGOTIATING" 
STATE_INPUT = "INPUT"             
STATE_FEEDBACK = "FEEDBACK"       
STATE_COUNTDOWN = "COUNTDOWN"     
STATE_AI_THINKING = "AI_THINKING" 


class Main:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()

        # Core Components
        self.world = World(screen, theme)
        self.creative_state = CreativeState()
        self.cocreative = CoCreativeManager(theme)

        # UI Components
        self.input_box = InputBox(WIDTH // 2 - 250, HEIGHT // 2 - 16, 500, 32)

        # State
        self.state = STATE_WAITING
        self.ai_proposal = None
        self.proposal_handled = False

        # Timers
        self.countdown_val = 3
        self.countdown_timer = 0

        self.pending_kind = None          
        self.pending_prompt = None       
        self.pending_params = None    

        self.thinking_text = ""
        self.modify_hint = None
        self.input_source = "user"    

    def main_loop(self):
        while True:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.update()
            self.draw()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # GLOBAL: open prompt box with SHIFT (only when not in modal states)
            if event.type == pygame.KEYDOWN:
                # Open Input Box (SHIFT) - Valid in most states except INPUT/NEGOTIATING/FEEDBACK/AI_THINKING
                if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT) and self.state not in (
                    STATE_INPUT, STATE_NEGOTIATING, STATE_FEEDBACK, STATE_AI_THINKING
                ):
                    self._enter_input_mode()
                    continue

            # State-specific input handlers
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
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self._restart_game_round()

            elif self.state == STATE_AI_THINKING:
        
                pass

    def update(self):
        msgs = self.cocreative.poll()
        for m in msgs:
            if m["kind"] == "parse_done":
                intent = m["intent"]
                prompt_text = m["text"]

                base_params = self.pending_params if self.pending_params is not None else self.creative_state.get_params()
                new_params = self.cocreative.apply_intent(prompt_text, intent, base_params)

                self._apply_new_level(prompt_text, new_params)

                self.pending_kind = None
                self.pending_prompt = None
                self.pending_params = None

            elif m["kind"] == "proposal_done":
                self.ai_proposal = m["proposal"]
                self.pending_kind = None
                self.state = STATE_NEGOTIATING
                print(f"Negotiating: {self.ai_proposal}")

            elif m["kind"] == "error":
                print("AI worker error:", m["error"])
                self.pending_kind = None
                self.state = STATE_GAME_OVER

        # Physics
        if self.state == STATE_PLAYING:
            self.world.update()
            if self.world.game_over:
                self._on_death()

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

        # Ground scroll
        self.world._scroll_world()

    def draw(self):
        self.screen.fill((0, 0, 0))

        bg = theme.get("background")
        if bg:
            self.screen.blit(pygame.transform.scale(bg, (WIDTH, HEIGHT)), (0, 0))

        self.world.draw()

        ground = theme.get("ground")
        if ground:
            self.screen.blit(ground, (self.world.world_shift, HEIGHT))

        # Overlays
        if self.state == STATE_GAME_OVER and not self.ai_proposal:
            self.world.game.instructions()

        if self.state == STATE_NEGOTIATING:
            self._draw_overlay((0, 0, 20, 200))
            self._draw_negotiation_ui()

        if self.state == STATE_FEEDBACK:
            self._draw_overlay((0, 0, 0, 220))
            self._draw_feedback_ui()

        if self.state == STATE_INPUT:
            self._draw_overlay((0, 0, 0, 150))
            self.input_box.draw(self.screen)

        if self.state == STATE_COUNTDOWN:
            self._draw_text(str(self.countdown_val), 80, WIDTH // 2, HEIGHT // 2, (255, 255, 255))

        if self.state == STATE_WAITING:
            self._draw_text("Press SPACE to start the game", 50, WIDTH // 2, HEIGHT // 2, (255, 255, 255))

        if self.state == STATE_PLAYING:
            snapshot = self.world.get_snapshot()
            if snapshot["score"] > 5:
                self._draw_text("Press SHIFT to have fun!!!", 40, WIDTH // 2, HEIGHT - 50, (255, 255, 0))

        if self.state == STATE_AI_THINKING:
            self._draw_overlay((0, 0, 0, 180))
            self._draw_text("AI thinking...", 40, WIDTH // 2, HEIGHT // 2, (255, 255, 255))

        pygame.display.update()

    # ==========================
    # LOGIC HELPERS
    # ==========================

    def _on_death(self):
        self.proposal_handled = False

        # If in Creative Mode, get feedback first
        if self.creative_state.is_active():
            self.state = STATE_FEEDBACK
        else:
            self.state = STATE_GAME_OVER
            self._check_for_ai_intervention()

    def _restart_game_round(self):
        self.world.reset_game_logic()
        self.world.set_physics_params(self.creative_state.current_params)
        self.state = STATE_WAITING

    def _check_for_ai_intervention(self):
        if self.proposal_handled:
            return
        self.proposal_handled = True

        snapshot = self.world.get_snapshot()
        score = snapshot["score"]

        if score >= 0:
            good_examples = self.creative_state.get_good_examples()
            self.pending_kind = "proposal"
            self.cocreative.request_proposal(score, 0, snapshot, good_examples)
            self.thinking_text = "Generating suggestion..."
            self.state = STATE_AI_THINKING

    def _apply_new_level(self, prompt, params):
        print("Committing new level...")
        self.creative_state.activate(params, prompt)
        self.world.reset_game_logic()
        self.world.set_physics_params(params)
        self.state = STATE_COUNTDOWN
        self.countdown_val = 3
        self.countdown_timer = pygame.time.get_ticks()

        self.ai_proposal = None
        self.modify_hint = None

    def _handle_feedback(self, event):
        if event.type == pygame.KEYDOWN:
            rating = None
            if event.key == pygame.K_1:
                rating = "good"
            elif event.key == pygame.K_2:
                rating = "okay"
            elif event.key == pygame.K_3:
                rating = "bad"

            if rating:
                self.creative_state.register_feedback(rating)
                try:
                    self.cocreative.record_feedback(rating)
                except Exception:
                    pass
                self.state = STATE_GAME_OVER
                self._check_for_ai_intervention()

    def _draw_feedback_ui(self):
        self._draw_text("HOW WAS THIS LEVEL?", 50, WIDTH // 2, HEIGHT // 2 - 80, (255, 255, 0))
        self._draw_text(f"Current: {self.creative_state.current_prompt}", 30, WIDTH // 2, HEIGHT // 2 - 30, (200, 200, 200))

        self._draw_text("[1] GOOD (Keep it)", 40, WIDTH // 2, HEIGHT // 2 + 40, (0, 255, 0))
        self._draw_text("[2] OKAY (Improve it)", 40, WIDTH // 2, HEIGHT // 2 + 90, (255, 200, 0))
        self._draw_text("[3] BAD (Reject it)", 40, WIDTH // 2, HEIGHT // 2 + 140, (255, 50, 50))

    def _handle_input(self, event):
        prompt = self.input_box.handle_event(event)

        if self.input_box.done:
            if prompt:
                if prompt.strip().lower() == "reset":
                    self.creative_state.reset()
                    self.world.reset_game_logic()
                    self.world.reset_physics_to_default()
                    self.state = STATE_WAITING
                else:
                    current = self.creative_state.get_params()
                    self.pending_kind = "parse"
                    self.pending_prompt = prompt
                    self.pending_params = current
                    self.cocreative.request_prompt(prompt, source=self.input_source)
                    self.thinking_text = f"Parsing: {prompt}"
                    self.state = STATE_AI_THINKING
            else:
                self.state = self.state_before_input
                self.state = STATE_COUNTDOWN
                self.countdown_val = 3
                self.countdown_timer = pygame.time.get_ticks()

            self.input_box.active = False

    def _enter_input_mode(self, text="", hint=None, return_state=None, source="user"):
        self.state_before_input = return_state if return_state is not None else self.state
        self.modify_hint = hint
        self.input_source = source

        self.state = STATE_INPUT
        self.input_box.reset()
        self.input_box.set_text(text or "")  
        self.input_box.active = True

    def _handle_negotiation(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_y:
                current = self.creative_state.get_params()
                proposal_text = self.ai_proposal or ""
                self.pending_kind = "parse"
                self.pending_prompt = proposal_text
                self.pending_params = current
                self.cocreative.request_prompt(proposal_text, source="ai_accept")
                self.thinking_text = f"Parsing: {proposal_text}"
                self.state = STATE_AI_THINKING

            elif event.key == pygame.K_n:
                try:
                    self.cocreative.record_reject(self.ai_proposal)
                except Exception:
                    pass
                self.state = STATE_GAME_OVER
                self.ai_proposal = None
                self.modify_hint = None
                self.world.game.instructions()

            elif event.key == pygame.K_m:
                base = self.ai_proposal or ""
                print(f"Modify chosen. Base suggestion: {base}")
                self.ai_proposal = None
                self._enter_input_mode(text="", hint=base, return_state=STATE_GAME_OVER, source="ai_modify")

    def _draw_negotiation_ui(self):
        self._draw_text("AI SUGGESTION", 50, WIDTH // 2, HEIGHT // 2 - 80, (0, 255, 255))
        self._draw_text(f"\"{self.ai_proposal}\"", 30, WIDTH // 2, HEIGHT // 2, (255, 255, 255))
        self._draw_text("[Y] Accept  [N] Reject  [M] Modify", 30, WIDTH // 2, HEIGHT // 2 + 60, (200, 200, 200))

    def _draw_overlay(self, color):
        s = pygame.Surface((WIDTH, HEIGHT + GROUND_HEIGHT))
        s.set_alpha(color[3])
        s.fill(color[:3])
        self.screen.blit(s, (0, 0))

    def _draw_text(self, text, size, x, y, color):
        font = pygame.font.Font(None, size)
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(x, y))
        self.screen.blit(surf, rect)


if __name__ == "__main__":
    game = Main(screen)
    soundPlay("background")
    game.main_loop()
