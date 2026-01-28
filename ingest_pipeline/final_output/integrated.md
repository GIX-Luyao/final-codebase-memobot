(venv) jasonyang@Jasons-MacBook-Pro-3 ingest_pipeline % python3 main.py
Processing 1 video(s) from /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/data

>>> Processing: 30s_clip.mp4
============================================================
Step 1: process_video (diarization, faces, face matching)
============================================================
[Pipeline] Processing video: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/data/30s_clip.mp4
[Info] Intermediate outputs will be saved to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1

=== Step 1: Extracting audio ===
[OK] Audio extracted and saved to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/audio/extracted_audio.wav
[OK] Audio uploaded: media://audio/224e1206e5924854bfd47a54d2a49c73.wav

=== Step 2: Speaker diarization ===
[OK] Found 8 speaker turns
  [001] SPEAKER_00 1.36-1.97s: Hi there.
  [002] SPEAKER_00 2.38-4.22s: I'm Jason. Nice to meet you.
  [003] SPEAKER_00 4.95-6.00s: Hi there, Mammobot.
  [004] SPEAKER_00 6.50-12.88s: Uh today I went to eat lunch and then I also had
  [005] SPEAKER_00 13.06-15.51s: some noodles for lunch today.
  [006] SPEAKER_00 17.39-19.59s: Today it's also raining outside
  [007] SPEAKER_00 19.98-24.61s: and I'm a little bit worried about the traffic on the way home.
  [008] SPEAKER_00 26.18-27.98s: So that's where I am today.
[OK] Diarization results saved to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/diarization.json

=== Step 2b: Speaker enrollment ===

[Enroll] Processing SPEAKER_00...
[Enroll] SPEAKER_00 -> person_89a155b3 (confidence: 95.0)
[OK] Mapped 1 speakers to voiceprints
  SPEAKER_00 -> person_89a155b3

=== Step 3: Extracting faces at speaker timestamps ===
[Info] Processing 8 unique timestamps (one per speaker turn start)
[TalkNet] Running: python3 demoTalkNet.py --videoName ingest_6cc1aac1 --confidenceThreshold -0.5
2026-01-28 13:58:54 Extract the video and save in demo/ingest_6cc1aac1/pyavi/video.avi 
2026-01-28 13:58:54 Extract the audio and save in demo/ingest_6cc1aac1/pyavi/audio.wav 
2026-01-28 13:58:55 Extract the frames and save in demo/ingest_6cc1aac1/pyframes 
VideoManager is deprecated and will be removed.
`base_timecode` argument is deprecated and has no effect.
demo/ingest_6cc1aac1/pyavi/video.avi - scenes detected 1
2026-01-28 13:58:56 Scene detection and save in demo/ingest_6cc1aac1/pywork 
2026-01-28 13:59:51 Face detection and save in demo/ingest_6cc1aac1/pywork 
2026-01-28 13:59:51 Face track and detected 1 tracks 
100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.28s/it]
2026-01-28 13:59:54 Face Crop and saved in demo/ingest_6cc1aac1/pycrop tracks 
01-28 13:59:54 Model para number = 15.01
Model pretrain_TalkSet.model loaded from previous state! 
100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:29<00:00, 29.42s/it]
2026-01-28 14:00:23 Scores extracted and saved in demo/ingest_6cc1aac1/pywork 
100%|██████████████████████████████████████████████████████████████████████████████████████| 727/727 [00:05<00:00, 129.97it/s]

[Timestamp 1/8] t=1.17s (speaker: SPEAKER_00)
[Detection] Found 1 green boxes, 0 red boxes (before NMS)
[Detection] After NMS: 1 boxes (filtered 0)
  [OK] Found 1 face boxes, speaker_box=found
  [OK] Saved outputs to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/faces/t001165

[Timestamp 2/8] t=2.18s (speaker: SPEAKER_00)
[Detection] Found 1 green boxes, 0 red boxes (before NMS)
[Detection] After NMS: 1 boxes (filtered 0)
  [OK] Found 1 face boxes, speaker_box=found
  [OK] Saved outputs to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/faces/t002184

[Timestamp 3/8] t=4.75s (speaker: SPEAKER_00)
[Detection] Found 1 green boxes, 0 red boxes (before NMS)
[Detection] After NMS: 1 boxes (filtered 0)
  [OK] Found 1 face boxes, speaker_box=found
  [OK] Saved outputs to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/faces/t004745

[Timestamp 4/8] t=6.30s (speaker: SPEAKER_00)
[Detection] Found 1 green boxes, 0 red boxes (before NMS)
[Detection] After NMS: 1 boxes (filtered 0)
  [OK] Found 1 face boxes, speaker_box=found
  [OK] Saved outputs to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/faces/t006305

[Timestamp 5/8] t=12.87s (speaker: SPEAKER_00)
[Detection] Found 1 green boxes, 0 red boxes (before NMS)
[Detection] After NMS: 1 boxes (filtered 0)
  [OK] Found 1 face boxes, speaker_box=found
  [OK] Saved outputs to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/faces/t012865

[Timestamp 6/8] t=17.19s (speaker: SPEAKER_00)
[Detection] Found 1 green boxes, 0 red boxes (before NMS)
[Detection] After NMS: 1 boxes (filtered 0)
  [OK] Found 1 face boxes, speaker_box=found
  [OK] Saved outputs to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/faces/t017185

[Timestamp 7/8] t=19.79s (speaker: SPEAKER_00)
[Detection] Found 1 green boxes, 0 red boxes (before NMS)
[Detection] After NMS: 1 boxes (filtered 0)
  [OK] Found 1 face boxes, speaker_box=found
  [OK] Saved outputs to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/faces/t019785

[Timestamp 8/8] t=25.98s (speaker: SPEAKER_00)
[Detection] Found 1 green boxes, 0 red boxes (before NMS)
[Detection] After NMS: 1 boxes (filtered 0)
  [OK] Found 1 face boxes, speaker_box=found
  [OK] Saved outputs to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/faces/t025985

=== Step 4: Matching 16 faces against database ===
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1769637631.031343 4408480 pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
I0000 00:00:1769637631.031424 4408480 pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
[Timing] Total face matching started at 14:00:32 for 16 faces
[Timing] Face recognition started at 14:00:32 for t1165_face_00_green.jpg
[Timing] Face recognition ended at 14:00:32 for t1165_face_00_green.jpg (took 0.049s)
[Match] t1165_face_00_green -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.5183241091165721)
[Timing] Face recognition started at 14:00:32 for t1165_speaker.jpg
[Timing] Face recognition ended at 14:00:32 for t1165_speaker.jpg (took 0.049s)
[Match] t1165_speaker -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.5183241091165721)
[Timing] Face recognition started at 14:00:32 for t2184_face_00_green.jpg
[Timing] Face recognition ended at 14:00:32 for t2184_face_00_green.jpg (took 0.045s)
[Match] t2184_face_00_green -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.4166970749738178)
[Timing] Face recognition started at 14:00:32 for t2184_speaker.jpg
[Timing] Face recognition ended at 14:00:32 for t2184_speaker.jpg (took 0.044s)
[Match] t2184_speaker -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.4166970749738178)
[Timing] Face recognition started at 14:00:32 for t4745_face_00_green.jpg
[Timing] Face recognition ended at 14:00:32 for t4745_face_00_green.jpg (took 0.046s)
[Match] t4745_face_00_green -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.4425113464904248)
[Timing] Face recognition started at 14:00:32 for t4745_speaker.jpg
[Timing] Face recognition ended at 14:00:32 for t4745_speaker.jpg (took 0.046s)
[Match] t4745_speaker -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.4425113464904248)
[Timing] Face recognition started at 14:00:32 for t6305_face_00_green.jpg
[Timing] Face recognition ended at 14:00:32 for t6305_face_00_green.jpg (took 0.049s)
[Match] t6305_face_00_green -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.36410953784705435)
[Timing] Face recognition started at 14:00:32 for t6305_speaker.jpg
[Timing] Face recognition ended at 14:00:32 for t6305_speaker.jpg (took 0.043s)
[Match] t6305_speaker -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.36410953784705435)
[Timing] Face recognition started at 14:00:32 for t12865_face_00_green.jpg
[Timing] Face recognition ended at 14:00:32 for t12865_face_00_green.jpg (took 0.044s)
[Match] t12865_face_00_green -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.40728803916561507)
[Timing] Face recognition started at 14:00:32 for t12865_speaker.jpg
[Timing] Face recognition ended at 14:00:32 for t12865_speaker.jpg (took 0.044s)
[Match] t12865_speaker -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.40728803916561507)
[Timing] Face recognition started at 14:00:32 for t17185_face_00_green.jpg
[Timing] Face recognition ended at 14:00:32 for t17185_face_00_green.jpg (took 0.043s)
[Match] t17185_face_00_green -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.37792627548354785)
[Timing] Face recognition started at 14:00:32 for t17185_speaker.jpg
[Timing] Face recognition ended at 14:00:32 for t17185_speaker.jpg (took 0.042s)
[Match] t17185_speaker -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.37792627548354785)
[Timing] Face recognition started at 14:00:32 for t19785_face_00_green.jpg
[Timing] Face recognition ended at 14:00:32 for t19785_face_00_green.jpg (took 0.042s)
[Match] t19785_face_00_green -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.40650101938909156)
[Timing] Face recognition started at 14:00:32 for t19785_speaker.jpg
[Timing] Face recognition ended at 14:00:32 for t19785_speaker.jpg (took 0.042s)
[Match] t19785_speaker -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.40650101938909156)
[Timing] Face recognition started at 14:00:32 for t25985_face_00_green.jpg
[Timing] Face recognition ended at 14:00:32 for t25985_face_00_green.jpg (took 0.042s)
[Match] t25985_face_00_green -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.37328787882914505)
[Timing] Face recognition started at 14:00:32 for t25985_speaker.jpg
[Timing] Face recognition ended at 14:00:32 for t25985_speaker.jpg (took 0.043s)
[Match] t25985_speaker -> 677b0814-3675-45b5-aef9-fcf64965ad5f (distance=0.37328787882914505)
[Timing] Total face matching ended at 14:00:32
[Timing] Total time for 16 face recognitions: 0.734s (avg: 0.046s per face)
[OK] Face matching results saved to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/face_matching.json

=== Step 5: Combining results ===

=== Step 6: Updating persons.db with speaker IDs ===
[OK] No updates needed in persons.db
[OK] Final results saved to: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1/final_results.json
Processed 8 speaker turns. Intermediate outputs: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/intermediate_outputs/run_6cc1aac1

============================================================
Step 2: vector_db (embeddings + Pegasus metadata → Pinecone)
============================================================
Processing video: 30s_clip
Uploading local video file: /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/data/30s_clip.mp4
Created asset: id=697a8711d0ad2fbb70afb8d2
Pegasus task created: id=697a8715bbe0cb5367281798
  Pegasus index status=pending
  Pegasus index status=indexing
  Pegasus index status=indexing
  Pegasus index status=indexing
  Pegasus index status=indexing
  Pegasus index status=indexing
  Pegasus index status=indexing
  Pegasus index status=ready
Pegasus upload complete. video_id=697a8715bbe0cb5367281798
Ingested 3 embeddings for /Users/jasonyang/Documents/Development/memobot/ingest_pipeline/data/30s_clip.mp4

Ingest complete.

============================================================
FINAL JSON OUTPUT
============================================================
[
  {
    "person_id": "ccad744a-17b9-4064-99db-855c87da4cf1",
    "name": "Jason",
    "clip_summary": "Jason, wearing glasses and a black puffer jacket, stands against a white wall with exposed pipes and ductwork, greeting the camera and discussing his lunch and concerns about traffic due to the rain.",
    "audio_dialogue": "Jason: \"Hi there.\"\nJason: \"I'm Jason. Nice to meet you.\"\nJason: \"Hi there, Mammobot.\"\nJason: \"Uh today I went to eat lunch and then I also had\"\nJason: \"some noodles for lunch today.\"\nJason: \"Today it's also raining outside\"\nJason: \"and I'm a little bit worried about the traffic on the way home.\"\nJason: \"So that's where I am today.\""
  }
]