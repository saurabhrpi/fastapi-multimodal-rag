import os
import requests
import yt_dlp
import openai
from dotenv import load_dotenv
from typing import List, Dict
import json
import time
import asyncio
#from pydub import AudioSegment
import subprocess

# Load API keys from .env
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with new syntax
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Milvus configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "in03-874be76b9aa0be7.serverless.gcp-us-west1.cloud.zilliz.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "268c3796886a41827afcee6560f083fbfc4992ae7265598b4d3582979748054380929293cd76ea79244845abf9773e4e9128de0e")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "youtubeCreaterVideos")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))

CHANNEL_USERNAMES = [
    "Eczachly_",
    "JoeReisData",
    "DannyMa"
]

MAX_VIDEOS = 20
CHUNK_LENGTH_SECONDS = 30 * 60  # 30 minutes


# Import MilvusVectorDB class from main.py
import sys
sys.path.append('.')
from main import MilvusVectorDB

# Initialize vector database
vector_db = MilvusVectorDB()


def get_channel_id(username: str) -> str:
    url = f"https://www.googleapis.com/youtube/v3/channels?part=id&forUsername={username}&key={YOUTUBE_API_KEY}"
    resp = requests.get(url)
    data = resp.json()
    if data.get("items"):
        return data["items"][0]["id"]
    # Try as custom URL (for channels without username)
    url = f"https://www.googleapis.com/youtube/v3/channels?part=id&forHandle={username}&key={YOUTUBE_API_KEY}"
    resp = requests.get(url)
    data = resp.json()
    if data.get("items"):
        return data["items"][0]["id"]
    raise ValueError(f"Channel not found for username: {username}")


def get_recent_short_videos(channel_id: str) -> List[Dict]:
    # Get uploads playlist
    url = f"https://www.googleapis.com/youtube/v3/channels?part=contentDetails&id={channel_id}&key={YOUTUBE_API_KEY}"
    resp = requests.get(url)
    uploads_playlist = resp.json()["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    # Get recent videos
    url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&playlistId={uploads_playlist}&maxResults={MAX_VIDEOS}&key={YOUTUBE_API_KEY}"
    resp = requests.get(url)
    items = resp.json().get("items", [])
    videos = []
    for item in items:
        video_id = item["snippet"]["resourceId"]["videoId"]
        title = item["snippet"]["title"]
        # Get video details for duration
        vurl = f"https://www.googleapis.com/youtube/v3/videos?part=contentDetails&id={video_id}&key={YOUTUBE_API_KEY}"
        vresp = requests.get(vurl)
        vdata = vresp.json()["items"][0]["contentDetails"]
        duration = parse_iso8601_duration(vdata["duration"])
        print(f"Duration: {duration}")
        print(f"Title: {title}")
        videos.append({
            "video_id": video_id,
            "title": title,
            "duration": duration
        })
        if len(videos) >= MAX_VIDEOS:
            break
    return videos


def parse_iso8601_duration(duration: str) -> int:
    import re
    match = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration
    )
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def download_audio(video_id: str, out_dir: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(out_dir, f'{video_id}.%(ext)s'),
        'quiet': True,
        'noplaylist': True,
        'extractaudio': True,
        'audioformat': 'mp3',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def transcribe_audio(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return transcript


async def clean_segment_with_llm(text: str) -> str:
    """
    Use OpenAI LLM to clean up transcript text while preserving specific details and concrete information.
    """
    print(f"DEBUG: Original transcript text: {text}")
    
    prompt = (
        "Clean up this transcript segment for clarity and grammar, but PRESERVE all specific details, numbers, names, and concrete information. "
        "Do not make it generic - keep the specific facts, figures, and details intact. "
        "Only remove filler words like 'um', 'uh', 'like', 'you know' if they don't add meaning. "
        "Return only the cleaned text.\nSegment: " + text
    )
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4", # Changed to gpt-4 for better cleaning
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3
        )
        cleaned_text = response.choices[0].message.content.strip()
        print(f"DEBUG: LLM cleaned text: {cleaned_text}")
        return cleaned_text
    except Exception as e:
        print(f"LLM cleaning failed, returning original text. Error: {e}")
        return text

def combine_segments(segments: list, min_duration: float = 10.0, min_chars: int = 50) -> list:
    """
    Combine adjacent segments into larger chunks for better RAG context.
    """
    if not segments:
        return []
    
    combined_chunks = []
    current_chunk = {
        "text": "",
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "segments": []
    }
    
    for segment in segments:
        # Add segment to current chunk
        current_chunk["text"] += " " + segment["text"]
        current_chunk["end"] = segment["end"]
        current_chunk["segments"].append(segment)
        
        # Check if we should finalize this chunk
        duration = current_chunk["end"] - current_chunk["start"]
        should_finalize = (
            duration >= min_duration and 
            len(current_chunk["text"].strip()) >= min_chars
        )
        
        if should_finalize:
            combined_chunks.append(current_chunk)
            # Start new chunk
            current_chunk = {
                "text": "",
                "start": segment["end"],
                "end": segment["end"],
                "segments": []
            }
    
    # Add final chunk if it has content
    if current_chunk["text"].strip():
        combined_chunks.append(current_chunk)
    
    return combined_chunks

async def process_transcript_for_rag(transcript_data: dict, video_metadata: dict) -> list[dict]:
    """
    Process transcript data into chunks suitable for RAG, cleaning each segment with LLM.
    Each chunk will be a segment with timing information.
    """
    chunks = []
    if "segments" in transcript_data:
        # Combine segments into larger chunks
        combined_segments = combine_segments(transcript_data["segments"])
        print(f"DEBUG: Combined {len(transcript_data['segments'])} segments into {len(combined_segments)} chunks")
        
        tasks = []
        for combined_segment in combined_segments:
            tasks.append(clean_segment_with_llm(combined_segment["text"].strip()))
        cleaned_texts = await asyncio.gather(*tasks)
        
        for i, (combined_segment, cleaned_text) in enumerate(zip(combined_segments, cleaned_texts)):
            chunk = {
                "content": cleaned_text,
                "metadata": {
                    "video_id": video_metadata["video_id"],
                    "video_title": video_metadata["title"],
                    "channel_name": video_metadata["channel"],
                    "start_time": combined_segment["start"],
                    "end_time": combined_segment["end"],
                    "duration": combined_segment["end"] - combined_segment["start"],
                    "segment_id": i,
                    "source_type": "youtube_transcript",
                    "timestamp": time.time()
                }
            }
            chunks.append(chunk)
    else:
        cleaned_text = await clean_segment_with_llm(transcript_data.get("text", ""))
        chunk = {
            "content": cleaned_text,
            "metadata": {
                "video_id": video_metadata["video_id"],
                "video_title": video_metadata["title"],
                "channel_name": video_metadata["channel"],
                "start_time": 0,
                "end_time": video_metadata.get("duration", 0),
                "duration": video_metadata.get("duration", 0),
                "segment_id": 0,
                "source_type": "youtube_transcript",
                "timestamp": time.time()
            }
        }
        chunks.append(chunk)
    return chunks


def add_transcript_to_rag(transcript_chunks: List[Dict]) -> int:
    """
    Add transcript chunks to the RAG system (Milvus vector database).
    Returns the number of chunks successfully added.
    """
    added_count = 0
    
    for chunk in transcript_chunks:
        try:
            print(f"DEBUG: Metadata being sent to Milvus: {chunk['metadata']}")
            success = vector_db.add_document(
                content=chunk["content"],
                metadata=chunk["metadata"]
            )
            if success:
                added_count += 1
                print(f"    ✅ Added chunk {chunk['metadata']['segment_id']} to RAG system")
            else:
                print(f"    ❌ Failed to add chunk {chunk['metadata']['segment_id']} to RAG system")
        except Exception as e:
            print(f"    ❌ Error adding chunk to RAG: {e}")
    
    return added_count


def split_audio_to_chunks(audio_path: str, chunk_length: int = CHUNK_LENGTH_SECONDS) -> list:
    import subprocess, os
    # Convert to wav first if not already wav
    base, ext = os.path.splitext(audio_path)
    if ext.lower() != ".wav":
        wav_path = f"{base}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True)
    else:
        wav_path = audio_path
    output_pattern = f"{base}_chunk_%03d.wav"
    cmd = [
        "ffmpeg",
        "-i", wav_path,
        "-f", "segment",
        "-segment_time", str(chunk_length),
        "-c:a", "pcm_s16le",
        output_pattern
    ]
    subprocess.run(cmd, check=True)
    chunk_paths = []
    idx = 0
    while True:
        chunk_file = f"{base}_chunk_{idx:03d}.wav"
        if os.path.exists(chunk_file):
            chunk_paths.append(chunk_file)
            idx += 1
        else:
            break
    return chunk_paths


def main():
    os.makedirs("downloads", exist_ok=True)
    os.makedirs("transcripts", exist_ok=True)
    
    total_videos_processed = 0
    total_chunks_added = 0
    
    for username in CHANNEL_USERNAMES:
        print(f"Processing channel: {username}")
        try:
            channel_id = get_channel_id(username)
            videos = get_recent_short_videos(channel_id)
            
            for video in videos:
                print(f"  Processing: {video['title']} ({video['duration']}s)")
                
                # Download audio
                audio_path = download_audio(video["video_id"], "downloads")
                print(f"    Audio downloaded: {audio_path}")
                
                # Split audio if needed
                chunk_paths = split_audio_to_chunks(audio_path, CHUNK_LENGTH_SECONDS)
                print(f"    Audio split into {len(chunk_paths)} chunk(s)")
                
                for idx, chunk_path in enumerate(chunk_paths):
                    # Transcribe audio chunk
                    print(f"    Transcribing chunk {idx+1}/{len(chunk_paths)}...")
                    transcript = transcribe_audio(chunk_path)
                    
                    # Save transcript to file
                    print(f"    Saving transcript to file...")
                    out_path = os.path.join("transcripts", f"{video['video_id']}_chunk_{idx}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(transcript.model_dump(), f, ensure_ascii=False, indent=2)
                    print(f"    Transcript saved: {out_path}")
                    
                    # Process transcript for RAG
                    video_metadata = {
                        "video_id": video["video_id"],
                        "title": video["title"],
                        "duration": video["duration"],
                        "channel": username
                    }
                    transcript_chunks = asyncio.run(process_transcript_for_rag(transcript.model_dump(), video_metadata))
                    print(f"    Created {len(transcript_chunks)} chunks for RAG")
                    
                    # Add to RAG system
                    chunks_added = add_transcript_to_rag(transcript_chunks)
                    total_chunks_added += chunks_added
                    print(f"    Added {chunks_added}/{len(transcript_chunks)} chunks to RAG system")
                    
                    # Clean up audio chunk file
                    try:
                        os.remove(chunk_path)
                        print(f"    Cleaned up audio chunk file")
                    except:
                        pass
                
                total_videos_processed += 1
                
                # Clean up original audio file
                try:
                    os.remove(audio_path)
                    print(f"    Cleaned up audio file")
                except:
                    pass
                
        except Exception as e:
            print(f"Error processing {username}: {e}")
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Videos processed: {total_videos_processed}")
    print(f"Chunks added to RAG: {total_chunks_added}")
    
    # Get RAG system stats
    try:
        stats = vector_db.get_stats()
        print(f"RAG system status: {stats['status']}")
        print(f"Total entities in collection: {stats.get('total_entities', 0)}")
    except Exception as e:
        print(f"Could not get RAG stats: {e}")


if __name__ == "__main__":
    main() 