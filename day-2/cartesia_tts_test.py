import os
import sounddevice as sd
import numpy as np
from cartesia import Cartesia

def main():
    # Configuration
    api_key = os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        raise ValueError("CARTESIA_API_KEY environment variable must be set")

    # Text to synthesize
    transcript = "Hi, this is a demo of the TTS, Text to Speech from Cartesia. It's one of the easiest ones to use, Eleven Lab is another example."

    # Voice configuration - using voice ID
    # Example voice ID (replace with your preferred voice)
    # You can get available voices from: https://docs.cartesia.ai/voices
    voice_id = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"  # Example voice

    # Initialize client with new API
    client = Cartesia(api_key=api_key)

    # Generate audio using the bytes() method with streaming
    print("Generating audio with Sonic-3...")

    # Configure output format
    output_format = {
        "container": "wav",
        "encoding": "pcm_f32le",
        "sample_rate": 44100
    }

    # Voice configuration with mode and ID
    voice_config = {
        "mode": "id",
        "id": voice_id
    }

    # Generate audio bytes (streaming)
    audio_chunks = []
    try:
        for chunk in client.tts.bytes(
            model_id="sonic-3",
            transcript=transcript,
            voice=voice_config,
            output_format=output_format,
            language="en"
        ):
            audio_chunks.append(chunk)

        print(f"Generated {len(audio_chunks)} audio chunks")

        # Combine all chunks
        audio_data = b"".join(audio_chunks)

        # Parse WAV data directly from bytes to play immediately
        print("Playing audio...")

        import io
        try:
            import soundfile as sf
            # Read directly from bytes buffer
            audio_buffer = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_buffer, dtype='float32')
            sd.play(audio_array, sample_rate, blocking=True)
            print("Playback complete!")
        except ImportError:
            # Fallback: use scipy.io.wavfile
            print("soundfile not found, trying scipy...")
            from scipy.io import wavfile
            audio_buffer = io.BytesIO(audio_data)
            sample_rate, audio_array = wavfile.read(audio_buffer)
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32) / 32768.0
            sd.play(audio_array, sample_rate, blocking=True)
            print("Playback complete!")

    except Exception as e:
        print(f"Error during TTS generation: {e}")
        raise

def demo_async():
    """
    Example of async usage (requires asyncio)
    """
    import asyncio
    from cartesia import AsyncCartesia

    async def generate_async():
        api_key = os.environ.get("CARTESIA_API_KEY")
        client = AsyncCartesia(api_key=api_key)

        voice_config = {
            "mode": "id",
            "id": "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"
        }

        output_format = {
            "container": "wav",
            "encoding": "pcm_f32le",
            "sample_rate": 44100
        }

        audio_chunks = []
        async for chunk in await client.tts.bytes(
            model_id="sonic-3",
            transcript="This is an async example!",
            voice=voice_config,
            output_format=output_format,
            language="en"
        ):
            audio_chunks.append(chunk)

        return b"".join(audio_chunks)

    # Uncomment to run async demo
    # asyncio.run(generate_async())

if __name__ == "__main__":
    main()

    # Uncomment to try async demo:
    # demo_async()
