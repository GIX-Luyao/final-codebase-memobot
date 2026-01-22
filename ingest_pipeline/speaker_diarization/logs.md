(venv) jasonyang@Jasons-MacBook-Pro-3 speaker_diarization % python3 enroll_from_local_wav.py videoplayb
ack.mp4

=== Turn-level transcription (diarization speakers) ===
[001] SPEAKER_00 1.19-1.58  Marie.
[002] SPEAKER_00 2.23-3.21  This is Ms. Novak.
[003] SPEAKER_00 3.90-5.04  She's from Chicago.
[004] SPEAKER_00 5.49-6.80  Marie is our receptionist.
[005] SPEAKER_01 7.04-9.43  It's nice to meet you. It's nice to meet you too.
[006] SPEAKER_00 9.48-10.51  I'll get your tickets.
[007] SPEAKER_01 17.46-19.66  You look very familiar to me, Marie.
[008] SPEAKER_01 20.11-21.14  Have we met before?
[009] SPEAKER_02 21.57-23.30  I don't think so, no.
[010] SPEAKER_01 24.12-26.07  Well, I never forget a face.
[011] SPEAKER_01 27.05-29.32  I'm sure I know you from somewhere.
[012] SPEAKER_01 30.73-32.01  I don't look familiar to you.
[013] SPEAKER_02 32.92-34.20  I'm sorry, no.
[014] SPEAKER_01 35.41-36.30  I know.
[015] SPEAKER_01 36.77-38.42  We met in Chicago.
[016] SPEAKER_01 38.73-42.62  You were a waitress in a restaurant near the Art Institute.
[017] SPEAKER_02 43.34-45.20  I've never been to Chicago.
[018] SPEAKER_01 45.34-45.66  Oh.
[019] SPEAKER_01 46.73-49.09  Have you ever driven a taxi in Egypt?
[020] SPEAKER_02 49.47-50.10  No
[021] SPEAKER_01 51.92-54.28  Oh, you were the pilot.
[022] SPEAKER_01 55.16-59.51  On a small airplane in China, you flew me over the Great Wall!
[023] SPEAKER_03 59.77-60.51  No.
[024] SPEAKER_01 62.87-65.44  Have you ever gone snorkeling in Australia?
[025] SPEAKER_01 67.31-67.78  No.
[026] SPEAKER_01 68.89-70.39  Driven a bus in Peru?
[027] SPEAKER_03 72.03-77.84  Ms. Novak, I'm quite sure we've never met before. I came here only a year ago from Paris.
[028] SPEAKER_01 78.58-79.12  Paris?
[029] SPEAKER_01 79.92-82.48  Well, my sister Katarina lived there for a year.
[030] SPEAKER_01 84.31-85.00  Katerina?
[031] SPEAKER_01 86.00-88.17  Katarina Novak? Yes.
[032] SPEAKER_01 89.05-90.19  She lived with me.
[033] SPEAKER_01 90.67-94.56  Of course, you were in all the pictures she sent home.
[034] SPEAKER_01 95.78-98.50  What a coincidence! You see?
[035] SPEAKER_01 99.06-101.20  I never forget a face.

--- Processing SPEAKER_00 ---
[enroll] SPEAKER_00 not identified -> creating new voiceprint 'person_588a2731'
[saved] voiceprints.json updated with person_588a2731

--- Processing SPEAKER_01 ---
[enroll] SPEAKER_01 not identified -> creating new voiceprint 'person_6dede9a3'
[saved] voiceprints.json updated with person_6dede9a3

--- Processing SPEAKER_02 ---
[match] SPEAKER_02 -> person_588a2731

--- Processing SPEAKER_03 ---
[match] SPEAKER_03 -> person_588a2731

=== Final turn-level transcript (with identities) ===
[001] person_588a2731 (SPEAKER_00) 1.19-1.58  Marie.
[002] person_588a2731 (SPEAKER_00) 2.23-3.21  This is Ms. Novak.
[003] person_588a2731 (SPEAKER_00) 3.90-5.04  She's from Chicago.
[004] person_588a2731 (SPEAKER_00) 5.49-6.80  Marie is our receptionist.
[005] person_6dede9a3 (SPEAKER_01) 7.04-9.43  It's nice to meet you. It's nice to meet you too.
[006] person_588a2731 (SPEAKER_00) 9.48-10.51  I'll get your tickets.
[007] person_6dede9a3 (SPEAKER_01) 17.46-19.66  You look very familiar to me, Marie.
[008] person_6dede9a3 (SPEAKER_01) 20.11-21.14  Have we met before?
[009] person_588a2731 (SPEAKER_02) 21.57-23.30  I don't think so, no.
[010] person_6dede9a3 (SPEAKER_01) 24.12-26.07  Well, I never forget a face.
[011] person_6dede9a3 (SPEAKER_01) 27.05-29.32  I'm sure I know you from somewhere.
[012] person_6dede9a3 (SPEAKER_01) 30.73-32.01  I don't look familiar to you.
[013] person_588a2731 (SPEAKER_02) 32.92-34.20  I'm sorry, no.
[014] person_6dede9a3 (SPEAKER_01) 35.41-36.30  I know.
[015] person_6dede9a3 (SPEAKER_01) 36.77-38.42  We met in Chicago.
[016] person_6dede9a3 (SPEAKER_01) 38.73-42.62  You were a waitress in a restaurant near the Art Institute.
[017] person_588a2731 (SPEAKER_02) 43.34-45.20  I've never been to Chicago.
[018] person_6dede9a3 (SPEAKER_01) 45.34-45.66  Oh.
[019] person_6dede9a3 (SPEAKER_01) 46.73-49.09  Have you ever driven a taxi in Egypt?
[020] person_588a2731 (SPEAKER_02) 49.47-50.10  No
[021] person_6dede9a3 (SPEAKER_01) 51.92-54.28  Oh, you were the pilot.
[022] person_6dede9a3 (SPEAKER_01) 55.16-59.51  On a small airplane in China, you flew me over the Great Wall!
[023] person_588a2731 (SPEAKER_03) 59.77-60.51  No.
[024] person_6dede9a3 (SPEAKER_01) 62.87-65.44  Have you ever gone snorkeling in Australia?
[025] person_6dede9a3 (SPEAKER_01) 67.31-67.78  No.
[026] person_6dede9a3 (SPEAKER_01) 68.89-70.39  Driven a bus in Peru?
[027] person_588a2731 (SPEAKER_03) 72.03-77.84  Ms. Novak, I'm quite sure we've never met before. I came here only a year ago from Paris.
[028] person_6dede9a3 (SPEAKER_01) 78.58-79.12  Paris?
[029] person_6dede9a3 (SPEAKER_01) 79.92-82.48  Well, my sister Katarina lived there for a year.
[030] person_6dede9a3 (SPEAKER_01) 84.31-85.00  Katerina?
[031] person_6dede9a3 (SPEAKER_01) 86.00-88.17  Katarina Novak? Yes.
[032] person_6dede9a3 (SPEAKER_01) 89.05-90.19  She lived with me.
[033] person_6dede9a3 (SPEAKER_01) 90.67-94.56  Of course, you were in all the pictures she sent home.
[034] person_6dede9a3 (SPEAKER_01) 95.78-98.50  What a coincidence! You see?
[035] person_6dede9a3 (SPEAKER_01) 99.06-101.20  I never forget a face.

Voiceprints stored at: state/voiceprints.json