#!/usr/bin/env python3
"""
so101_keyboard_teleop.py  –  Fixed version
Changes vs original:
  1. IK joint extraction is validated against actual chain link count so a
     mismatched URDF doesn't silently swap joints.
  2. Joint limits are clamped to the actuator ranges from the XML before
     sending to the bridge.
  3. Gripper toggle is debounced so a single SPACE press doesn't flicker.
  4. Added R/F keys for wrist pitch (was described in print but missing).
  5. UDP send is skipped if IK returned NaN (happens when target is
     unreachable) instead of crashing the bridge.
"""

import os
import json
import math
import socket
import threading
import time

import numpy as np

try:
    import ikpy.chain
except ImportError:
    print('ikpy is not installed.  Run: pip install ikpy')
    raise SystemExit(1)

try:
    from pynput import keyboard
except ImportError:
    print('pynput is not installed.  Run: pip install pynput')
    raise SystemExit(1)


# ── joint limits matching the MuJoCo XML actuator ctrlrange ─────────────────
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
JOINT_LIMITS = [
    (-1.91986,  1.91986),   # shoulder_pan
    (-1.74533,  1.74533),   # shoulder_lift
    (-1.69000,  1.69000),   # elbow_flex
    (-1.65806,  1.65806),   # wrist_flex
    (-2.74385,  2.84121),   # wrist_roll
    (-0.17453,  1.74533),   # gripper
]

JOINT_NAMES = [
    'shoulder_pan', 'shoulder_lift', 'elbow_flex',
    'wrist_flex', 'wrist_roll', 'gripper',
]
# ────────────────────────────────────────────────────────────────────────────


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class KeyboardTeleopNode:
    def __init__(self):
        print('Setting up Keyboard Teleop Node …')

        self.sock                = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM)
        self.mujoco_bridge_addr  = ('127.0.0.1', 9876)

        # ── URDF / IK chain ─────────────────────────────────────────────────
        urdf_path = 'src/so101_description/urdf/so101.urdf'
        if not os.path.exists(urdf_path):
            print(f'ERROR: URDF not found at {urdf_path}')
            raise SystemExit(1)

        # active_links_mask must have exactly one True per controlled joint.
        # [False=base, True×5=arm, False=ee] → 7 entries → ik_joints has 7 values
        # Arm joints live at indices 1-5 (0=base, 6=ee are inactive).
        active_mask = [False, True, True, True, True, True, False]
        self.ik_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path, active_links_mask=active_mask)

        n_links = len(self.ik_chain.links)
        # Validate indices we're about to use
        expected_n = len(active_mask)
        if n_links != expected_n:
            print(
                f'WARNING: IK chain has {n_links} links but mask has '
                f'{expected_n} entries.  Check your URDF.  '
                f'Attempting to continue …')

        # We extract indices 1-5 from the IK solution (0=base, 6=ee)
        self._ik_arm_indices = list(range(1, 6))   # [1,2,3,4,5]

        self.prev_joints = [0.0] * n_links

        # ── state ────────────────────────────────────────────────────────────
        # Safe starting position above the table, well within reach
        self.target_pos   = np.array([0.20, 0.0, 0.15])
        self.speed        = 0.003          # metres per 50 Hz tick
        self.spin_speed   = 0.05           # radians per 50 Hz tick

        self.wrist_roll_offset  = 0.0
        self.wrist_pitch_offset = 0.0

        self.gripper_open   = True         # True=open(1.74), False=closed(0)
        self._space_pressed = False        # debounce flag

        self.pressed_keys = set()
        self.running      = True

        self._loop_thread = threading.Thread(
            target=self._ik_loop, daemon=True)
        self._loop_thread.start()

    # ── public entry point ───────────────────────────────────────────────────

    def start(self):
        print()
        print('=' * 60)
        print('  KEYBOARD TELEOP  –  SO-101')
        print('=' * 60)
        print('  Arrow UP / DOWN   : Forward / Backward  (X axis)')
        print('  Arrow LEFT/RIGHT  : Left / Right         (Y axis)')
        print('  W / S             : Up / Down            (Z axis)')
        print('  Q / E             : Wrist roll CCW / CW')
        print('  R / F             : Wrist pitch up / down')
        print('  SPACEBAR          : Toggle gripper open / close')
        print('  ESC               : Quit')
        print('=' * 60)
        print()

        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()

        self.running = False
        print('Keyboard listener stopped.')

    # ── keyboard callbacks ───────────────────────────────────────────────────

    def on_press(self, key):
        self.pressed_keys.add(key)

        # Gripper toggle — debounced: only fire on the first press event
        if key == keyboard.Key.space and not self._space_pressed:
            self._space_pressed = True
            self.gripper_open   = not self.gripper_open

        if key == keyboard.Key.esc:
            self.running = False
            return False   # stop listener

    def on_release(self, key):
        self.pressed_keys.discard(key)
        if key == keyboard.Key.space:
            self._space_pressed = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _chars(self) -> set:
        """Return the set of lowercase character keys currently held."""
        result = set()
        for k in self.pressed_keys:
            try:
                if k.char:
                    result.add(k.char.lower())
            except AttributeError:
                pass
        return result

    def _update_velocity(self):
        chars = self._chars()
        dx, dy, dz = 0.0, 0.0, 0.0

        if keyboard.Key.up    in self.pressed_keys: dx += 1.0
        if keyboard.Key.down  in self.pressed_keys: dx -= 1.0
        if keyboard.Key.left  in self.pressed_keys: dy += 1.0
        if keyboard.Key.right in self.pressed_keys: dy -= 1.0
        if 'w' in chars: dz += 1.0
        if 's' in chars: dz -= 1.0

        # Wrist orientation overrides
        if 'q' in chars: self.wrist_roll_offset  -= self.spin_speed
        if 'e' in chars: self.wrist_roll_offset  += self.spin_speed
        if 'r' in chars: self.wrist_pitch_offset += self.spin_speed
        if 'f' in chars: self.wrist_pitch_offset -= self.spin_speed

        return dx, dy, dz

    # ── main IK loop ─────────────────────────────────────────────────────────

    def _ik_loop(self):
        while self.running:
            dx, dy, dz = self._update_velocity()

            # Move the Cartesian target
            self.target_pos[0] += dx * self.speed
            self.target_pos[1] += dy * self.speed
            self.target_pos[2] += dz * self.speed

            # Clamp to a reachable workspace box
            self.target_pos[0] = clamp(self.target_pos[0], 0.05,  0.40)
            self.target_pos[1] = clamp(self.target_pos[1], -0.35, 0.35)
            self.target_pos[2] = clamp(self.target_pos[2], -0.05, 0.45)

            # Clamp wrist offsets to their joint limits
            lo_r, hi_r = JOINT_LIMITS[4]   # wrist_roll
            lo_p, hi_p = JOINT_LIMITS[3]   # wrist_flex (pitch)
            self.wrist_roll_offset  = clamp(
                self.wrist_roll_offset,  lo_r, hi_r)
            self.wrist_pitch_offset = clamp(
                self.wrist_pitch_offset, lo_p, hi_p)

            # Solve IK
            ik_joints = self.ik_chain.inverse_kinematics(
                target_position=self.target_pos,
                initial_position=self.prev_joints,
            )

            # Guard against NaN / Inf from an unreachable target
            if not np.all(np.isfinite(ik_joints)):
                time.sleep(0.02)
                continue

            self.prev_joints = list(ik_joints)

            # Extract the 5 arm joints at indices 1-5
            # Index:  0=base(skip), 1=shoulder_pan, 2=shoulder_lift,
            #         3=elbow_flex, 4=wrist_flex,   5=wrist_roll, 6=ee(skip)
            try:
                arm_vals = [float(ik_joints[i]) for i in self._ik_arm_indices]
            except IndexError as exc:
                print(f'IK index error: {exc}  chain has '
                      f'{len(ik_joints)} values, '
                      f'expected indices {self._ik_arm_indices}')
                time.sleep(0.02)
                continue

            # Apply manual wrist overrides
            # arm_vals[3] = wrist_flex (index 4 in chain, position 3 in list)
            # arm_vals[4] = wrist_roll (index 5 in chain, position 4 in list)
            arm_vals[3] += self.wrist_pitch_offset
            arm_vals[4] += self.wrist_roll_offset

            # Gripper value
            gripper_val = JOINT_LIMITS[5][1] if self.gripper_open \
                else JOINT_LIMITS[5][0] + 0.05   # slight close margin

            # Build full payload  [pan, lift, flex, wrist_flex, roll, gripper]
            payload = arm_vals + [gripper_val]

            # Clamp every joint to its hardware limit before sending
            payload = [
                clamp(v, lo, hi)
                for v, (lo, hi) in zip(payload, JOINT_LIMITS)
            ]

            # Send — skip if still NaN after all the above
            if all(math.isfinite(v) for v in payload):
                self.sock.sendto(
                    json.dumps(payload).encode('utf-8'),
                    self.mujoco_bridge_addr)

            time.sleep(0.02)   # 50 Hz


if __name__ == '__main__':
    node = KeyboardTeleopNode()
    try:
        node.start()
    except KeyboardInterrupt:
        pass