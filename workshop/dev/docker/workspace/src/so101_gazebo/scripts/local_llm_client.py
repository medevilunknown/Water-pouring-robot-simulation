#!/usr/bin/env python3
"""
Local LLM Client (Ollama)
=========================
Handles communication with the local Ollama instance for planning
grasps and diagnosing failures.
"""

import json
import logging
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class LocalLLMClient:
    def __init__(self, model='llama3.2', host='http://host.docker.internal:11434'):
        self.model = model
        self.logger = logging.getLogger('LocalLLMClient')
        self.enabled = False
        if not OLLAMA_AVAILABLE:
            self.logger.warning("Ollama library not installed. LLM features disabled.")
            return
        try:
            import httpx
            self.client = ollama.Client(host=host, timeout=httpx.Timeout(5.0))
            # Pre-flight: test if server is reachable
            self.client.list()
            self.enabled = True
            self.logger.info("Ollama server is reachable. LLM features enabled.")
        except Exception as e:
            self.logger.warning(f"Ollama server unreachable ({e}). LLM features disabled — using scripted fallback.")

        
    def _query(self, system_prompt, user_prompt, response_format=None):
        if not self.enabled:
            return "{}" if response_format else ""
            
        try:
            options = {'temperature': 0.1}
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
            kwargs = {'model': self.model, 'messages': messages, 'options': options}
            if response_format:
                kwargs['format'] = response_format
                
            response = self.client.chat(**kwargs)
            return response['message']['content']
        except Exception as e:
            self.logger.error(f"Ollama query failed: {e}")
            return "{}" if response_format else ""

    def diagnose_failure(self, episode_log):
        """Diagnose failure and suggest config changes."""
        system_prompt = """You are an expert robotics engineer diagnosing a failed pouring task in Gazebo.
You will be provided with an episode log containing telemetry, parameters, and the point of failure.

Analyze the failure and suggest adjustments to the configuration parameters to fix the issue.
Available parameters to adjust:
- approach_offset_z: Vertical offset for pre-grasp (m)
- grip_position: Target position for the gripper joint (-0.2 to 1.2)
- pour_tilt_deg: Angle to tilt the bottle (degrees)
- pour_duration_sec: How long to hold the pour (seconds)

Output MUST be valid JSON containing ONLY the suggested config adjustments, like this:
{
    "grip_position": -0.15,
    "approach_offset_z": 0.05
}
Do not include any explanations or markdown formatting outside the JSON object."""
        user_prompt = f"Episode Log Data:\n{json.dumps(episode_log, indent=2)}\n\nProvide the adjusted configuration."
        
        response = self._query(system_prompt, user_prompt, response_format='json')
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse LLM diagnosis output: {response}")
            return {}

    def plan_grasp(self, objects, current_config):
        """Plan grasp strategy."""
        system_prompt = """You are deciding the high-level actions for a robot arm.
Based on the detected objects, reply with 'PROCEED' to attempt a grasp and pour, or 'ABORT' if required objects are missing."""
        user_prompt = f"Detected Objects:\n{json.dumps(objects, indent=2)}\n\nConfig:\n{json.dumps(current_config, indent=2)}"
        
        response = self._query(system_prompt, user_prompt)
        if not response:
            self.logger.warning("LLM returned empty or failed. Defaulting to PROCEED.")
            return 'PROCEED'
            
        if 'PROCEED' in response.upper():
            return 'PROCEED'
        return 'ABORT'

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    client = LocalLLMClient()
    print("Testing connection (requires ollama server running)...")
    try:
        if OLLAMA_AVAILABLE:
            res = client.plan_grasp({"water_bottle": {"pose": [0,0,0]}}, {})
            print(f"Result: {res}")
    except Exception as e:
        print(f"Test failed: {e}")
