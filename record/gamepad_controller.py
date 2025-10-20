#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gamepad controller module for triggering recording controls.
Reads control_bindings.json and monitors gamepad inputs to trigger
start_recording, stop_recording, and emergency_stop actions.
"""
import json
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

try:
    from inputs import get_gamepad, devices
    INPUTS_AVAILABLE = True
except ImportError:
    INPUTS_AVAILABLE = False
    print("⚠️  Warning: 'inputs' library not available. Gamepad control disabled.")
    print("   Install with: pip install inputs")


class GamepadController:
    """
    Monitors gamepad inputs and triggers callbacks based on control_bindings.json
    """

    def __init__(self, bindings_file: str = 'control_bindings.json'):
        """
        Initialize gamepad controller.

        Args:
            bindings_file: Path to control_bindings.json file
        """
        self.bindings_file = Path(bindings_file)
        self.bindings = None
        self.recording_controls = {}
        self.callbacks = {}
        self.running = False
        self.thread = None

        # Track axis states to detect transitions
        self.axis_states = {}

        # Load bindings
        if not self._load_bindings():
            raise ValueError(f"Failed to load bindings from {bindings_file}")

        # Check if gamepad is available
        if not INPUTS_AVAILABLE:
            raise RuntimeError("inputs library not available")

        if not devices.gamepads:
            raise RuntimeError("No gamepad detected")

    def _load_bindings(self) -> bool:
        """Load control bindings from JSON file."""
        try:
            with open(self.bindings_file, 'r') as f:
                data = json.load(f)
                self.bindings = data.get('bindings', {})
                self.recording_controls = self.bindings.get('recording_controls', {})

            if not self.recording_controls:
                print("⚠️  Warning: No recording controls found in bindings file")
                return False

            print(f"✅ Loaded control bindings from {self.bindings_file}")
            print(f"   Recording controls: {list(self.recording_controls.keys())}")
            return True

        except FileNotFoundError:
            print(f"❌ Bindings file not found: {self.bindings_file}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in bindings file: {e}")
            return False

    def register_callback(self, control_name: str, callback: Callable):
        """
        Register a callback function for a control.

        Args:
            control_name: Name of the control (e.g., 'start_recording', 'stop_recording')
            callback: Function to call when control is triggered (no arguments)
        """
        if control_name not in self.recording_controls:
            print(f"⚠️  Warning: Control '{control_name}' not found in bindings")
            return

        self.callbacks[control_name] = callback
        print(f"✅ Registered callback for '{control_name}'")

    def _check_trigger(self, control_name: str, event) -> bool:
        """
        Check if an event triggers the specified control.

        Args:
            control_name: Name of the control to check
            event: Input event from gamepad

        Returns:
            True if the control was triggered
        """
        if control_name not in self.recording_controls:
            return False

        binding = self.recording_controls[control_name]
        axis_code = binding.get('axis')
        trigger_value = binding.get('value')

        # Button press (Key event)
        if event.ev_type == 'Key' and event.code == axis_code:
            # Trigger on button press (state == 1)
            return event.state == 1

        # Axis/switch trigger (Absolute event)
        elif event.ev_type == 'Absolute' and event.code == axis_code:
            # Track previous state
            prev_state = self.axis_states.get(axis_code, 0)
            current_state = event.state
            self.axis_states[axis_code] = current_state

            # Trigger on transition TO the trigger value
            # Use a tolerance for analog axes
            tolerance = 100  # Allow ±100 deviation

            was_inactive = abs(prev_state - trigger_value) > tolerance
            is_active = abs(current_state - trigger_value) <= tolerance

            return was_inactive and is_active

        return False

    def _gamepad_loop(self):
        """Main loop that monitors gamepad inputs (runs in separate thread)."""
        print("🎮 Gamepad monitoring started")
        print("   Waiting for control inputs...")

        while self.running:
            try:
                events = get_gamepad()
                for event in events:
                    # Check each registered control
                    for control_name, callback in self.callbacks.items():
                        if self._check_trigger(control_name, event):
                            print(f"🎮 Triggered: {control_name}")
                            # Call the callback in a try-except to prevent crashes
                            try:
                                callback()
                            except Exception as e:
                                print(f"❌ Error in callback for '{control_name}': {e}")

            except Exception as e:
                # Ignore occasional input errors
                if self.running:  # Only log if we're still supposed to be running
                    time.sleep(0.01)

        print("🎮 Gamepad monitoring stopped")

    def start(self):
        """Start monitoring gamepad inputs in a background thread."""
        if self.running:
            print("⚠️  Gamepad controller already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._gamepad_loop, daemon=True)
        self.thread.start()
        print("✅ Gamepad controller started")

    def stop(self):
        """Stop monitoring gamepad inputs."""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("✅ Gamepad controller stopped")

    def get_control_info(self, control_name: str) -> Optional[Dict]:
        """Get binding information for a specific control."""
        return self.recording_controls.get(control_name)

    def list_controls(self) -> Dict:
        """List all available recording controls."""
        return self.recording_controls.copy()


# Example usage
if __name__ == '__main__':
    import sys

    def on_start():
        print("▶️  START RECORDING triggered!")

    def on_stop():
        print("⏹️  STOP RECORDING triggered!")

    def on_emergency():
        print("🛑 EMERGENCY STOP triggered!")

    try:
        # Create controller
        controller = GamepadController()

        # Register callbacks
        controller.register_callback('start_recording', on_start)
        controller.register_callback('stop_recording', on_stop)
        controller.register_callback('emergency_stop', on_emergency)

        # Start monitoring
        controller.start()

        print("\n" + "="*60)
        print("Gamepad Controller Test")
        print("="*60)
        print("Press the configured buttons on your gamepad:")
        print("  - Start recording button")
        print("  - Stop recording button")
        print("  - Emergency stop button")
        print("\nPress Ctrl+C to exit")
        print("="*60 + "\n")

        # Keep running until interrupted
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if 'controller' in locals():
            controller.stop()
        sys.exit(0)
