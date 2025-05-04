import streamlit as st
import whisper
from moviepy.editor import VideoFileClip
from keybert import KeyBERT
import os

st.set_page_config(page_title="AI Video Highlighter", layout="centered")

st.title("🎬 AI-Powered Video Highlighter")
st.write("Upload a video or paste a YouTube link. This app will generate highlight clips with virality scores.")

# 📂 Output folder
OUTPUT_FOLDER = "highlight_clips"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file:
    with open("input_video.mp4", "wb") as f:
        f.write(video_file.read())
    VIDEO_PATH = "input_video.mp4"

    st.video(VIDEO_PATH)
    
    if st.button("Generate Highlights"):
        with st.spinner("🔍 Transcribing video..."):
            model = whisper.load_model("base")
            result = model.transcribe(VIDEO_PATH)
        
        st.success("Transcription complete.")
        
        # Extract keywords
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
        
        video = VideoFileClip(VIDEO_PATH)
        clips = []
        
        clip_duration = 60
        for seg in ranked_segments:
            start = seg["start"]
            end = min(start + clip_duration, video.duration)
            if end - start < 10:
                continue
            clip = video.subclip(start, end)
            clips.append((clip, seg))
            if len(clips) >= 3:
                break
        
        st.write("## 🎉 Top Highlight Clips:")
        
        for idx, (clip, seg) in enumerate(clips):
            output_file = os.path.join(OUTPUT_FOLDER, f"highlight_{idx+1}.mp4")
            clip.write_videofile(output_file, codec="libx264", audio_codec="aac")
            
            virality_score = min(100, round(seg["score"] / ranked_segments[0]["score"] * 100))
            
            st.video(output_file)
            st.write(f"**Clip {idx+1}:** Virality Score: {virality_score}/100")
            with open(output_file, "rb") as f:
                st.download_button(label="Download Clip", data=f, file_name=f"highlight_{idx+1}.mp4", mime="video/mp4")
        
        st.success("✅ Highlights generated!")

