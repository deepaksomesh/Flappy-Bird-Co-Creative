import pygame
from settings import DEFAULT_THEME

from settings import resource_path

class ThemeManager:
  def __init__(self):
    self.themes = {}
    self._load_themes()
    self.current_theme = DEFAULT_THEME

  def _load_themes(self):
    # Day
    self.themes["day"] = {
      "background": pygame.image.load(resource_path("assets/layout/day/bg.png")).convert(),
      "pipe": pygame.image.load(resource_path("assets/layout/day/pipe.png")).convert_alpha(),
      "ground": pygame.image.load(resource_path("assets/layout/day/ground.png")).convert(),
    }
    # Night
    self.themes["night"] = {
      "background": pygame.image.load(resource_path("assets/layout/night/bg_night.png")).convert(),
      "pipe": pygame.image.load(resource_path("assets/layout/night/pipe_night.png")).convert_alpha(),
      "ground": pygame.image.load(resource_path("assets/layout/night/ground_night.png")).convert(),
    }
    # Hell
    self.themes["hell"] = {
      "background": pygame.image.load(resource_path("assets/layout/hell/bg.png")).convert(),
      "pipe": pygame.image.load(resource_path("assets/layout/hell/pipe.png")).convert_alpha(),
      "ground": pygame.image.load(resource_path("assets/layout/hell/ground.png")).convert(),
    }
    # Space
    self.themes["space"] = {
      "background": pygame.image.load(resource_path("assets/layout/space/bg.png")).convert(),
      "pipe": pygame.image.load(resource_path("assets/layout/space/pipe.png")).convert_alpha(),
      "ground": pygame.image.load(resource_path("assets/layout/space/ground.png")).convert(),
    }

  def set_theme(self, theme):
    self.current_theme = theme
  
  def get_current_theme(self):
    return self.current_theme

  def get(self, asset_name):
    return self.themes[self.current_theme][asset_name]