# sound.py
import pygame

from settings import resource_path

pygame.mixer.init()

sounds = {
    "jump": pygame.mixer.Sound(resource_path("assets/sounds/jump.wav")),
    "score": pygame.mixer.Sound(resource_path("assets/sounds/score.wav")),
    "hit": pygame.mixer.Sound(resource_path("assets/sounds/hit.mp3")),
    "background": pygame.mixer.Sound(resource_path("assets/sounds/background.mp3")),
    "night": pygame.mixer.Sound(resource_path("assets/sounds/night.mp3")),
    "day": pygame.mixer.Sound(resource_path("assets/sounds/day.mp3")),
    "space": pygame.mixer.Sound(resource_path("assets/sounds/space.mp3")),
    "hell": pygame.mixer.Sound(resource_path("assets/sounds/hell.mp3")),
    "next": pygame.mixer.Sound(resource_path("assets/sounds/nextlevel.mp3")),
}

def play(sound_name):
    if sound_name in sounds:
        sounds[sound_name].play()

def stop(sound_name):
    if sound_name in sounds:
        sounds[sound_name].stop()