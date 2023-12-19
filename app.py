import gradio as gr
import soundfile as sf
from faster_whisper import WhisperModel

model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

def transcribe_audio(audio):
    sr, data = audio
    temp_file = "temp.wav"
    sf.write(temp_file, data, sr, format='wav')
    segments, info = model.transcribe(temp_file)
    result = ""
    for segment in segments:
        result += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
    return result

iface = gr.Interface(
    fn=transcribe_audio,
    inputs=["microphone"],
    outputs=gr.Textbox(),
    title="Team UNDERGOD SIF Hackathon Audio to Text Demo (Press Submit Again if it shows Error!)",
    description="This Demo Shows our state of the art solution for Psuedo real-time audio transcription (Only English Accepted)"
)

iface.launch(debug=True)
