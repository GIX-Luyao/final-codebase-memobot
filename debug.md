# server

use tcp server to transmit audio and image steams on different threads with corresponding ports


# record

audio jitter, use buffer to smooth
headphone not working
fix echoing by dialing down input audio when robot is speaking
concat user audio and robot audio by adding them together

treat the TTS audio as a stream.
If a new chunk arrives immediately after the previous one (e.g., within 200ms), ignore its timestamp and snap it to the end of the previous chunk. This creates a perfect, seamless sentence.
If a chunk arrives after a long pause (e.g., > 200ms), it's a new sentence. Respect the timestamp and leave the silence gap.




# face recognition
don't use 1st frame because camera is just turned on. use frame from 0.5s later


# memory retrieve

filter memory based on person id

