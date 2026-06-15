import asyncio
import os
import winsound
import pygame.mixer

class AudioManager:
    def __init__(self, audio_base_dir="audio"):
        pygame.mixer.init()
        pygame.mixer.set_num_channels(32)  # Increase channels for multiple sounds
        
        script_dir = os.path.dirname(__file__)
        self.full_audio_base_dir = os.path.join(script_dir, audio_base_dir)

        self.main_audio_channel = pygame.mixer.Channel(0)
        self.active_main_audio_file = None
        self.camera_prox_channels = {}

    async def play_beep(self, frequency=1000, duration=500):
        """Plays a simple beep sound."""
        # winsound.Beep is blocking, so run it in a separate thread
        await asyncio.to_thread(winsound.Beep, frequency, duration)

    async def play_calibration_beeps(self):
        """Plays a sequence of beeps for calibration feedback."""
        for _ in range(3):
            await self.play_beep(1500, 500)
            await asyncio.sleep(0.5)

    def play_looping_sound(self, filename):
        """Plays a sound on the main channel, looping indefinitely."""
        if self.active_main_audio_file != filename:
            self.stop_looping_sound()  # Stop current sound if different
            try:
                full_path = os.path.join(self.full_audio_base_dir, filename)
                sound = pygame.mixer.Sound(full_path)
                self.main_audio_channel.play(sound, -1)  # -1 loops
                self.active_main_audio_file = filename
            except pygame.error as e:
                print(f"Error playing audio file {full_path}: {e}")
                self.active_main_audio_file = None
            except FileNotFoundError:
                print(f"Error: Audio file not found at {full_path}")
                self.active_main_audio_file = None

    def stop_looping_sound(self):
        """Stops the sound playing on the main channel."""
        if self.main_audio_channel and self.main_audio_channel.get_busy():
            self.main_audio_channel.stop()
        self.active_main_audio_file = None

    def play_camera_prox_sound(self, marker_id, filename):
        """Plays a unique sound for a marker getting close to the camera."""
        if marker_id not in self.camera_prox_channels:
            try:
                full_path = os.path.join(self.full_audio_base_dir, filename)
                sound = pygame.mixer.Sound(full_path)
                channel = pygame.mixer.find_channel(True)  # Find an available channel
                if channel:
                    channel.play(sound, -1)
                    self.camera_prox_channels[marker_id] = channel
            except Exception as e:
                print(f"Error playing camera proximity sound for {marker_id}: {e}")

    def stop_camera_prox_sound(self, marker_id):
        """Stops the proximity sound for a specific marker."""
        if marker_id in self.camera_prox_channels:
            channel = self.camera_prox_channels.pop(marker_id)
            channel.stop()

    def stop_all_sounds(self):
        """Stops all currently playing sounds."""
        self.stop_looping_sound()
        for channel in self.camera_prox_channels.values():
            channel.stop()
        self.camera_prox_channels.clear()
        print("Stopped all audio channels.")
        
    def quit(self):
        """Quits the pygame mixer."""
        pygame.mixer.quit()
