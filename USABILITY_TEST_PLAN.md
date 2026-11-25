# Usability test plan for the Communicator (do this before 28 Nov)

**Goal:** show with real people that the single-switch Communicator works, and measure how fast it is.
This is a *usability* check with able-bodied volunteers using a single switch. It is not a clinical trial and says nothing about patients.

## Who and how many
5 volunteers (family, friends, classmates). Ideally also show it to one speech-language therapist or a local disability NGO and ask for feedback.
Get verbal consent, tell people what is recorded (only times and mistakes, no names), and let them stop at any time.

## Setup
Laptop or phone, Chrome, the Communicator page, scan speed 1.2 s, language of the volunteer's choice. Switch = Space bar (or tap). If you can borrow an adaptive USB switch, test that too.

## Tasks (read them out; time each one)
1. Select WATER and speak it.
2. Build the sentence “FOOD · TIRED” and speak it.
3. Trigger HELP and confirm the emergency phrase is spoken.
4. Select a phrase by mistake, then fix it with Undo.
5. Spell the word “WATER” using SPELL (note how many selections the suggestions save).
6. Report pain: PAIN → CHEST → 7, then speak it.
7. (Caregiver) add the custom phrase “Please turn on the fan” and select it.

Use “Build log” on the Communicator page after each volunteer and paste the CSV into your notes; it records the time of every selection, undos, and spelling speed (characters per minute).
Try “Auto speed” on and off, and “Ignore double press” at 0.5 s with one volunteer who presses twice by accident.

## Optional: sensor switches (needs a laptop with camera/microphone, Chrome, localhost or https)
In the Communicator choose “Sound” (clap) or “Camera motion” (head nod, brow raise, hand wave), press Calibrate and stay still for 2 s, then repeat tasks 1 and 3. Record: false triggers per minute (while idle), missed triggers, and selection time. Adjust Sensitivity and Camera area. These detectors were tested on synthetic signals only, so real-world numbers are your evidence.

## What to record (table)
| Volunteer | Task | Time to complete (s) | Errors | Comment |
|---|---|---|---|---|
Also copy the app's own “AVG TIME / SELECTION” after each person.

## Report (put one paragraph in the submission)
“N volunteers completed M tasks; mean selection time X s (range a–b); Y errors; most common feedback: …; changes we made: …”.
Be honest about the small sample and that no patients were involved.

## Optional second round
Try 0.8 s and 1.8 s scan speeds on 2 volunteers and report the speed/error trade-off.
