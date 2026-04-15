#!/usr/bin/env python3
import json
import os
from datetime import datetime

class EpisodeLogger:
    def __init__(self, log_dir="/tmp/pour_agent_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_episode = {}
        
    def start_episode(self, episode_num, config):
        self.current_episode = {
            "episode": episode_num,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "detections": {},
            "phases_completed": [],
            "failure_phase": None,
            "failure_reason": None,
            "water_poured_ml": 0.0,
            "llm_suggested_config": {},
            "success": False
        }
        
    def log_detection(self, name, data):
        self.current_episode["detections"][name] = data
        
    def log_phase(self, phase):
        self.current_episode["phases_completed"].append(phase)
        
    def log_failure(self, phase, reason):
        self.current_episode["failure_phase"] = phase
        self.current_episode["failure_reason"] = reason
        self.current_episode["success"] = False
        
    def log_success(self, water_poured):
        self.current_episode["water_poured_ml"] = water_poured
        self.current_episode["success"] = True
        
    def save(self):
        ep_num = self.current_episode.get("episode", 0)
        filename = os.path.join(self.log_dir, f"episode_{ep_num:03d}.json")
        with open(filename, 'w') as f:
            json.dump(self.current_episode, f, indent=2)
        return self.current_episode
