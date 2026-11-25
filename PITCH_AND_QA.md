# Pitch, demo video and judge Q&A

## 90-second pitch (speak it, don't read it)
1. **Hook (10 s):** “Every year India records over a million new strokes. Many survivors cannot speak or move, and WHO says nearly a billion people worldwide who need assistive technology are denied access.”
2. **Idea (15 s):** “NeuroPilot AI is an open, low-cost platform that turns brain-signal patterns into assistive commands, with a safety-first loop that adapts to each user.”
3. **Demo (40 s):** open the app → Communicator: press Space to scan, pick WATER, it speaks (try Hindi) → press HELP, the alarm sounds and a caregiver message is ready → then Devices: press 🧠 FORWARD, the wheelchair moves → raise noise, a low-confidence prediction is rejected. Or Judge mode. (SSVEP Speller types WATER → Devices: press 🧠 FORWARD, the wheelchair moves → raise noise, low-confidence prediction is rejected → Calibrate, accuracy recovers.
4. **Honesty (10 s):** “Signals are simulated. We are validating on public real-EEG data next, and nothing here is a medical device.”
5. **Close (15 s):** “By 2036, personal BCIs could be affordable. We are building the trustworthy open tool to learn and test them.”

## 3-minute demo video storyboard
| Time | Screen | Say |
|---|---|---|
| 0:00 | Impact 2036 | the problem, the access gap, one sentence of solution |
| 0:20 | Communicator | single-switch scanning, speak in Hindi, HELP alarm and caregiver link; “the switch can be a key, a blink sensor or a BCI command” |
| 0:50 | Live Pipeline → Start | signal → filter → features → classifier → action → feedback; latencies are measured |
| 0:50 | Signal Studio | raw vs filtered; notch removes 50 Hz |
| 1:10 | SSVEP Speller | type WATER; show ITR; raise noise; show errors honestly |
| 1:40 | Devices | 🧠 buttons drive hand and wheelchair; show E-STOP |
| 2:05 | Closed-Loop | raise noise, rejections appear, Calibrate, before/after |
| 2:30 | Architecture → Live decoder network | explainability: real weights |
| 2:45 | Settings → PDF report (backend) | evidence; honesty slide: simulated, next steps |

## Likely judge questions and honest answers
0. **What does it actually do today?** It lets a person communicate needs and call for help with one switch, in English or Hindi. The switch, speech output and alert link are real; the brain-signal input is simulated. Usability result: <fill in from the test plan>.
1. **Is this real brain data?** The BCI demo uses simulated signals. The pipeline was also validated on real clinical EEG (Bonn dataset, seizure detection about 97% seizure-vs-rest, cross-validated by recording; see the Real EEG page), but that is a different task from BCI commands. A script to evaluate on PhysioNet motor-imagery data is included; here is the result: <fill in>. Even then, accuracy from 1-second windows and 4 channels is modest.
2. **94% accuracy sounds too good.** It is accuracy on synthetic, separable patterns. It proves the pipeline works end to end, not that thoughts can be decoded.
3. **Why not use existing BCI systems?** I am not replacing clinical systems. This is an open, inexpensive tool for learning and prototyping, with explainability and safety built in.
4. **What does adaptation actually do?** Verified examples are added and the model is retrained. In tests, when field noise was higher than lab calibration, accuracy rose (e.g. 66% to about 82% at 12 µV). With an already robust model the gain can be small.
5. **How is safety handled?** Confidence threshold with rejection, E-STOP, no link to real devices, photosensitivity warning for flicker.
6. **Privacy?** EEG is sensitive. The browser version processes everything locally; the backend stores only what you create. Consent and security are requirements for any real deployment.
7. **Who is the user?** First students, researchers and NGOs prototyping assistive ideas; later, clinician-supervised pilots.
8. **Business model?** Early stage: open-source core; possible paid training and hosted research tooling. Do not overclaim.
9. **What is hardest to scale?** Real-EEG variability between people and sessions; that is why calibration and per-user adaptation are central.
10. **What did you build alone and with which tools?** Answer truthfully, including AI-assisted development, in line with the event rules.
11. **Next 6 months?** Run real-data evaluation, test with open EEG hardware, add a clinician and user advisory loop, write an ethics plan.

## Claims to avoid
“Reads thoughts”, “clinically validated”, “medical-grade”, “real-time guaranteed latency”, any accuracy figure without saying it is simulated.

## Plan to the deadline (today is Sat 21 Nov; closes Mon 30 Nov)
- **21–22 Nov:** run the real-data scripts on your laptop (README), paste results into SUBMISSION.md and Impact notes.
- **23 Nov:** `docker compose up --build` on your machine; fix any issue.
- **24–25 Nov:** record the 3-minute video; rehearse the 90-second pitch twice.
- **26–27 Nov:** fill the registration form; check the hackathon rules on AI-assisted work and team/solo; proofread.
- **28–29 Nov:** buffer; submit early. Do not wait for the last hour.
- Evaluation is 1–2 Dec; finalists are announced 5 Dec; Demo Day in Indore is 14 Dec.
