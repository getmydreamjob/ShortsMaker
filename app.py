import streamlit as st
import whisper
from keybert import KeyBERT
import subprocess
import os

st.set_page_config(page_title="AI Video Highlighter", layout="centered")

st.title("🎬 AI-Powered Video Highlighter")
st.write("Upload a video file. This app will generate highlight clips with virality scores.")

OUTPUT_FOLDER = "highlight_clips"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

def cut_video_ffmpeg(input_file, start, duration, output_file):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", input_file,
        "-ss", str(start),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        output_file,
        "-y"
    ]
    subprocess.run(cmd, check=True)

if video_file:
    input_video_path = "input_video.mp4"
    with open(input_video_path, "wb") as f:
        f.write(video_file.read())
    
    st.video(input_video_path)

    if st.button("Generate Highlights"):
        with st.spinner("🔍 Transcribing video..."):
            model = whisper.load_model("base")
            result = model.transcribe(input_video_path)
        
        st.success("✅ Transcription complete.")

        # Keyword extraction
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(result["text"], keyphrase_ngram_range=(1,2), stop_words='english', top_n=10)
        keyword_list = [kw[0].lower() for kw in keywords]

        st.write(f"**Top Keywords:** {', '.join(keyword_list)}")

        segments = result["segments"]
        ranked_segments = []
        for seg in segments:
            text = seg["text"].lower()
            keyword_hits = sum([text.count(kw) for kw in keyword_list])
            speech_density = len(text.split()) / (seg["end"] - seg["start"] + 1e-6)
            score = keyword_hits + 0.1 * speech_density
            ranked_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "keyword_hits": keyword_hits,
                "speech_density": speech_density,
                "score": score
            })

        ranked_segments.sort(key=lambda x: x["score"], reverse=True)

        st.write("## 🎉 Generated Highlight Clips:")
        
        clip_duration = 60  # seconds
        num_clips = 3
        clip_count = 0
        
        for idx, seg in enumerate(ranked_segments):
            if clip_count >= num_clips:
                break
            start_time = seg["start"]
            end_time = min(start_time + clip_duration, seg["end"])
            if end_time - start_time < 10:
                continue
            output_file = os.path.join(OUTPUT_FOLDER, f"highlight_{clip_count+1}.mp4")
            try:
                cut_video_ffmpeg(input_video_path, start_time, end_time - start_time, output_file)
                virality_score = min(100, round(seg["score"] / ranked_segments[0]["score"] * 100))
                st.video(output_file)
                st.write(f"**Clip {clip_count+1}:** Virality Score: {virality_score}/100")
                with open(output_file, "rb") as f:
                    st.download_button(label="Download Clip", data=f, file_name=f"highlight_{clip_count+1}.mp4", mime="video/mp4")
                clip_count += 1
            except Exception as e:
                st.error(f"❌ Failed to cut clip: {e}")

        if clip_count == 0:
            st.warning("No suitable highlights found.")

