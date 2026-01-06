import pygame

class Pipe(pygame.sprite.Sprite):
    def __init__(self, pos, width, height, flip, theme, assets, move_speed=0):
        super().__init__()
        self.width = width
        self.image = assets.get("pipe")
        self.image = pygame.transform.scale(self.image, (width, height))
        self.is_flipped = flip
        
        if self.is_flipped:
            flipped_image = pygame.transform.flip(self.image, False, True)
            self.image = flipped_image
            
        self.rect = self.image.get_rect(topleft = pos)
        self.mask = pygame.mask.from_surface(self.image)
        
        # Moving Pipe Logic
        self.move_speed = move_speed
        self.start_y = pos[1]
        self.move_dir = 1
        self.move_range = 60

    def get_is_flipped(self):
        return self.is_flipped

    def update(self, x_shift):
        # Horizontal Scroll
        self.rect.x += x_shift
        
        # Vertical Movement
        if self.move_speed > 0:
            self.rect.y += self.move_speed * self.move_dir
            
            # Bounce logic
            if abs(self.rect.y - self.start_y) > self.move_range:
                self.move_dir *= -1
        
        # Cleanup
        if self.rect.right < 0:
            self.kill()

    def update_theme(self, assets):
      self.image = assets.get("pipe")  
      self.image = pygame.transform.scale(self.image, (self.width, self.rect.height))
      
      if self.is_flipped:
          self.image = pygame.transform.flip(self.image, False, True)

      self.mask = pygame.mask.from_surface(self.image)