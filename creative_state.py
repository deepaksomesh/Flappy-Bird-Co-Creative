import copy
import json
import os

class CreativeState:
    """
    Layer 2: Creative State
    Responsible for:
    - Remembering AI-designed levels
    - Tracking current creative parameters
    - Holding feedback memory (Persisted to JSON)
    - Knowing whether creative mode is active
    - Session-persistent state
    """
    def __init__(self):
        # State Machine for Creative Mode
        # "INACTIVE", "ACTIVE", "ACTIVE_WITH_HISTORY"
        self.mode = "INACTIVE"
        
        # Default Parameters (The invariants)
        self.default_params = {
            "speed": -6.0,
            "gravity": 0.5,
            "vel": 10.0,
            "gap_multiplier": 1.0,
            "dist": 4,
            "game_mode": "day",
            "pipe_move_speed": 0.0
        }
        
        # Current Parameters (Active)
        self.current_params = copy.deepcopy(self.default_params)
        
        # Track the prompt that generated current params
        self.current_prompt = "Default"
        
        # History of changes (Stack of parameter sets)
        self.history = []
        
        # Feedback Memory (Persisted)
        self.feedback_file = "session_feedback.json"
        self.feedback_memory = self._load_from_disk()

    def _load_from_disk(self):
        """Load persistent feedback history"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                    print(f"Loaded {len(data)} feedback entries.")
                    return data
            except Exception as e:
                print(f"Error loading history: {e}")
                return []
        return []

    def _save_to_disk(self):
        """Save feedback history for research analysis"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_memory, f, indent=4)
            print("Session Data Saved to session_feedback.json")
        except Exception as e:
            print(f"Error saving history: {e}")

    def is_active(self):
        return self.mode != "INACTIVE"

    def activate(self, params, prompt="AI Suggestion"):
        self._push_history()
        self.current_params = params
        self.current_prompt = prompt
        if self.mode == "INACTIVE":
            self.mode = "ACTIVE"
        else:
            self.mode = "ACTIVE_WITH_HISTORY"
        print(f"Creative State ACTIVATED: {self.mode}")

    def reset(self):
        """triggered by users typing 'reset'"""
        self.current_params = copy.deepcopy(self.default_params)
        self.current_prompt = "Default"
        self.history.clear()
        self.mode = "INACTIVE"
        print("Creative State RESET to Defaults.")

    def _push_history(self):
        # Limit history size if needed, practically infinite for session
        self.history.append(copy.deepcopy(self.current_params))

    def get_params(self):
        return self.current_params

    def register_feedback(self, rating):
        """
        Store input/output pair with rating for CURRENT params.
        rating: "good", "okay", "bad"
        """
        entry = {
            "prompt": self.current_prompt,
            "params": copy.deepcopy(self.current_params),
            "rating": rating
        }
        self.feedback_memory.append(entry)
        self._save_to_disk() # Auto-save!
        print(f"Feedback Registered: {rating.upper()} for '{self.current_prompt}'")

    def get_good_examples(self):
        """ Returns list of prompts rated 'good' """
        return [f"'{x['prompt']}'" for x in self.feedback_memory if x['rating'] == 'good']

    def get_bad_params(self):
        """ Returns list of parameter dicts rated 'bad' """
        return [x['params'] for x in self.feedback_memory if x['rating'] == 'bad']
