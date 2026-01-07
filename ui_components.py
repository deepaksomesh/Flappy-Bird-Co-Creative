import pygame

class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.text = text
        self.placeholder = "Design your next level"
        self.font = pygame.font.Font(None, 32)
        self.txt_surface = self.font.render(text, True, self.color)
        self.active = False
        self.done = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.done = True
                    return self.text
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    self.done = True
                    return None
                else:
                    self.text += event.unicode
        return None

    def update(self):
        pass

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 0, 0), self.rect)
        
        max_width = self.rect.width - 20 # Padding
        words = self.text.split(' ') if self.text else self.placeholder.split(' ')
        lines = []
        current_line = []
        
        text_color = self.color if self.text else pygame.Color('gray50')
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            fw, fh = self.font.size(test_line)
            
            if fw < max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
                    
        if current_line:
            lines.append(' '.join(current_line))
        
        y_offset = self.rect.y + 5
        for line in lines:
            line_surf = self.font.render(line, True, text_color)
            text_rect = line_surf.get_rect(center=(self.rect.centerx, y_offset + 10))
            screen.blit(line_surf, text_rect)
            y_offset += 25 
            
        self.rect.h = max(32, y_offset - self.rect.y + 10)
        pygame.draw.rect(screen, self.color, self.rect, 2)

    def reset(self):
        self.text = ''
        self.active = True
        self.done = False

    def set_text(self, text):
        self.text = text
        self.update()
