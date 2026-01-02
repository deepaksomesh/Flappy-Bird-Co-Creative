import pygame
import sys
from settings import WIDTH, HEIGHT, GROUND_HEIGHT
from world import World
from theme import ThemeManager
from sound import play as soundPlay
from game import GameIndicator

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
pygame.display.set_caption("Super Flappy Bird")
theme = ThemeManager()


class Main:
    def __init__(self, screen):
      self.screen = screen
      self.ground_scroll = 0
      self.scroll_speed = -6
      self.FPS = pygame.time.Clock()
      self.stop_ground_scroll = False
      
      # UI & Co-Creative
      from ui_components import InputBox
      from cocreative import CoCreativeManager
      
      self.input_box = InputBox(WIDTH // 2 - 100, HEIGHT // 2 - 16, 200, 32)
      self.cocreative_manager = CoCreativeManager(theme)
      self.input_mode = False
      self.countdown_active = False
      self.countdown_start = 0
      self.countdown_value = 3

    def main(self):
      world = World(screen, theme, isMulti=True)
      
      # AI Proposal State
      ai_proposal = None
      negotiating = False
      proposal_handled = False

      while True:
        self.stop_ground_scroll = world.game_over

        # --- Draw background ---
        bg_img = pygame.transform.scale(theme.get('background'), (WIDTH, HEIGHT))
        self.screen.blit(bg_img, (0, 0))

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # --- 1. NEGOTIATION MODE ---
            if negotiating and ai_proposal:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y: # YES -> Accept
                        print(f"User Accepted: {ai_proposal}")
                        # Apply change
                        try:
                            # 1. Get current params
                            current_params = world.get_params()
                            # 2. Process new prompt (generates assets + new params)
                            new_params = self.cocreative_manager.process_prompt(ai_proposal, current_params)
                            # 3. Apply to world
                            world.update_dynamics(new_params)
                            # 4. Restart BUT keep Creative Mode active
                            world.update("restart")
                            # 5. Re-apply dynamics because restart might have reset them
                            world.update_dynamics(new_params)
                            
                        except Exception as e:
                            print(f"Error applying proposal: {e}")
                        negotiating = False
                        ai_proposal = None
                        
                    elif event.key == pygame.K_n: # NO -> Reject
                        print("User Rejected.")
                        negotiating = False
                        ai_proposal = None
                        # Just Game Over state proceeds naturally
                        
                    elif event.key == pygame.K_m or event.key == pygame.K_LSHIFT: # MODIFY -> Edit
                        self.input_mode = True
                        self.input_box.reset()
                        self.input_box.set_text(ai_proposal) # Use new method
                        self.input_box.active = True
                        negotiating = False
                        ai_proposal = None
                continue

            # --- 2. INPUT MODE ---
            if self.input_mode:
                prompt = self.input_box.handle_event(event)
                if self.input_box.done:
                    if prompt: # Enter pressed with text
                         print(f"User Prompt: {prompt}")
                         try:
                             current_params = world.get_params()
                             new_params = self.cocreative_manager.process_prompt(prompt, current_params)
                             world.update_dynamics(new_params)
                             if world.game_over: 
                                 world.update("restart")
                                 world.update_dynamics(new_params) # Re-apply after restart
                         except Exception as e:
                             print(f"Error processing prompt: {e}")
                    
                    # Exit input mode, start countdown
                    self.input_mode = False
                    self.input_box.active = False
                    self.countdown_active = True
                    self.countdown_value = 3
                    self.countdown_timer = pygame.time.get_ticks()
                    
                continue 

            if event.type == pygame.KEYDOWN:
                if self.countdown_active: continue 

                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    self.input_mode = True
                    self.input_box.reset()
                    self.input_box.active = True
                    continue

                if not world.playing and not world.game_over:
                  world.playing = True           
                if event.key == pygame.K_SPACE:
                  soundPlay("jump")
                  world.update("jump")
                if event.key == pygame.K_r:
                  world.update("restart")
                  world.creative_mode = False 
                  proposal_handled = False # Reset for new game 
        
        # --- Update game world ---
        if not self.input_mode and not self.countdown_active and not negotiating:
            world.update()
        world.draw()

        # --- GAME OVER / AI PROPOSAL ---
        if world.game_over:
          # Trigger Proposal ONCE when game ends
          if not negotiating and not ai_proposal and not proposal_handled: 
               # Simple logic: Trigger negotiation randomly or based on score
               if world.last_score >= 0: 
                   # Generate metrics roughly
                   time_alive = pygame.time.get_ticks() // 1000 # Mock time
                   ai_proposal = self.cocreative_manager.generate_proposal(world.last_score, time_alive)
                   negotiating = True
                   proposal_handled = True # Mark as handled for this death
          
          if not negotiating:
              game = GameIndicator(screen, theme)
              game.instructions()

        # --- Draw ground ---
        ground_img = theme.get('ground')
        self.screen.blit(ground_img, (self.ground_scroll, HEIGHT))

        if not self.stop_ground_scroll and not self.input_mode and not self.countdown_active and not negotiating:
            self.ground_scroll += self.scroll_speed
            if abs(self.ground_scroll) > 35:
                self.ground_scroll = 0
        
        # Draw UI (Input)
        if self.input_mode:
            s = pygame.Surface((WIDTH, HEIGHT+GROUND_HEIGHT))
            s.set_alpha(150)
            s.fill((0,0,0))
            self.screen.blit(s, (0,0))
            self.input_box.update()
            self.input_box.draw(self.screen)

        # Draw UI (Negotiation)
        if negotiating and ai_proposal:
            # Overlay
            s = pygame.Surface((WIDTH, HEIGHT+GROUND_HEIGHT))
            s.set_alpha(200)
            s.fill((0,0,20)) # Dark Blue tint for AI thought
            self.screen.blit(s, (0,0))
            
            font = pygame.font.Font(None, 40)
            
            # Title
            title = font.render("AI SUGGESTION", True, (0, 255, 255))
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 100))
            
            # Proposal Text (Wrapped)
            # Simple word wrap
            words = ai_proposal.split(' ')
            lines = []
            current_line = []
            for word in words:
                test_line = ' '.join(current_line + [word])
                if font.size(test_line)[0] < WIDTH - 40:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            lines.append(' '.join(current_line))
            
            y_offset = HEIGHT//2 - 40
            for line in lines:
                prop_text = font.render(line, True, (255, 255, 255))
                self.screen.blit(prop_text, (WIDTH//2 - prop_text.get_width()//2, y_offset))
                y_offset += 40
            
            # Instructions
            help_font = pygame.font.Font(None, 30)
            help_text = help_font.render("[Y] Accept  [N] Reject  [M] Modify", True, (200, 200, 200))
            self.screen.blit(help_text, (WIDTH//2 - help_text.get_width()//2, y_offset + 50))

        # Draw Countdown
        if self.countdown_active:
             current_time = pygame.time.get_ticks()
             if current_time - self.countdown_timer > 1000:
                 self.countdown_value -= 1
                 self.countdown_timer = current_time
                 if self.countdown_value <= 0:
                     self.countdown_active = False
             
             if self.countdown_active:
                 font = pygame.font.Font(None, 74)
                 text = font.render(str(self.countdown_value), True, (255, 255, 255))
                 text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
                 # Draw text with outline
                 outline = font.render(str(self.countdown_value), True, (0, 0, 0))
                 outline_rect = outline.get_rect(center=(WIDTH // 2, HEIGHT // 2))
                 self.screen.blit(outline, (outline_rect.x - 2, outline_rect.y))
                 self.screen.blit(outline, (outline_rect.x + 2, outline_rect.y))
                 self.screen.blit(outline, (outline_rect.x, outline_rect.y - 2))
                 self.screen.blit(outline, (outline_rect.x, outline_rect.y + 2))
                 self.screen.blit(text, text_rect)

        pygame.display.update()

        self.FPS.tick(60)

if __name__ == "__main__":
    play = Main(screen)
    soundPlay("background")
    play.main()
