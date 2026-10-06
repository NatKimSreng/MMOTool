"""
DAI Dubber — the app's code, one file per area:

  core         paths, Flask app, settings & API keys, history
  tasks        background tasks (parallel, pause / resume, saved), GPU turn-taking, Tasks API
  media        subtitle files, timing, ffmpeg (mix, merge, mux, burn subtitles, Demucs)
  asr          speech to text
  translate    translation to spoken Khmer
  voices       Khmer voices, voice conversion, cloned voices, TTS, speech track
  characters   who speaks each line
  record       re-voice a dub with your own recordings
  pipeline     the main dub job (with checkpoints)
  batch_queue  overnight queue
  tools        cut, subtitles only, SRT → speech, download, Studio export, batch merge
  cleanup      uploads cleanup
  pages        pages, files, folders, settings routes

app.py (next to this folder) imports them all and starts the server.
"""
