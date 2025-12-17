from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import uuid
import torch

from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
from piano_transcription_inference.utilities import RegressionPostProcessor, write_events_to_midi

app = FastAPI()

# --- CONFIGURATION ---
# 0.5 = Default (Strict)
# 0.2 = Sensitive (Catches soft notes)
# 0.1 = Very Sensitive (Catches breath/noise)
SENSITIVITY_THRESHOLD = 0.2

# Check for GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading ByteDance Model on {DEVICE}...")

# Initialize Model
transcriptor = PianoTranscription(device=DEVICE, checkpoint_path=None)
print("✅ Model Loaded!")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    input_filename = f"{uuid.uuid4()}_{audio.filename}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    output_midi_path = os.path.splitext(input_path)[0] + ".mid"

    try:
        # 1. Save Audio
        with open(input_path, "wb") as f:
            f.write(await audio.read())

        print(f"🎵 Transcribing {input_filename}...")

        # 2. Load Audio
        (audio_data, _) = load_audio(input_path, sr=sample_rate, mono=True)

        # 3. Run Model (writes a default midi too, but we overwrite with our custom post-processing)
        transcribed_dict = transcriptor.transcribe(audio_data, output_midi_path)
        output_dict = transcribed_dict["output_dict"]

        # 4. CUSTOM POST-PROCESSING (thresholds are set here)
        post_processor = RegressionPostProcessor(
            frames_per_second=100,
            classes_num=88,
            onset_threshold=SENSITIVITY_THRESHOLD,
            offset_threshold=0.5,
            frame_threshold=0.5,
            pedal_offset_threshold=0.2,
        )

        # IMPORTANT: in your installed version this returns (note_events, pedal_events)
        est_note_events, est_pedal_events = post_processor.output_dict_to_midi_events(
            output_dict
        )

        # 5. Write MIDI (your installed version expects pedal_events too)
        write_events_to_midi(
            0,  # start_time
            est_note_events,
            est_pedal_events,
            output_midi_path,  # midi_path
        )

        return FileResponse(
            output_midi_path,
            media_type="audio/midi",
            filename="transcribed_sensitive.mid",
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}

    finally:
        # Clean up uploaded audio file
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass
