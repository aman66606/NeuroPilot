# BUILD 2036 – Submission text (paste-ready, edit to your voice)

**Project:** NeuroPilot AI – Turning Neural Signals into Intelligent Actions
**Challenge:** Health, Longevity & Biotechnology
**Live demo:** <paste your published link>  ·  **Video:** <paste link>  ·  **Code:** <paste repo link>

## One-line pitch
An open, low-cost BCI research and education platform that turns EEG-style signals into assistive commands (communication board, wheelchair, cursor) through a safety-first closed loop that adapts to each user.

## Specific task it performs today
Let a person who cannot speak or use their hands **tell caregivers what they need (water, food, washroom, pain, medicine) and call for help**, in English or Hindi, using **one switch**. The Communicator page does single-switch row-column scanning over a main board (HELP is in the first row), a pain-assessment flow (where, then how strong 1–10), and a spelling board with word prediction for anything not on the board; caregivers can add custom phrases. It adapts its scan speed to errors, can ignore accidental double presses, offers audio cues for low vision, and speaks the sentence with the browser's voices, and an emergency phrase repeats loudly and opens a prefilled WhatsApp or dial link for a caregiver. Any on/off signal can be the switch: a key, a tap, an adaptive USB switch, a **sound (clap or voice)**, **camera motion** (head nod, brow raise, hand wave; no model download, processed locally), or a BCI command. In this prototype the BCI command is simulated; the switch, speech and alert are real.

## Problem
People who lose the ability to move or speak, for example after a stroke, need assistive technology, but access is limited and unequal.
- India records roughly 1.25–1.3 million new stroke cases a year (Lancet Neurology 2021; GBD 2019 via the Lancet SEAR series).
- WHO and UNICEF (2022) estimate that more than 2.5 billion people need assistive products and nearly 1 billion are denied access, with access as low as 3% of need in the poorest countries.
These figures show the size of the access gap, not the size of a BCI market. Brain-computer interfaces are one promising route for people with severe paralysis, but research platforms are expensive, closed or hard to learn.

## Solution
NeuroPilot AI is a complete, runnable platform for learning, prototyping and testing BCI ideas:
1. **Acquire** 4-channel EEG-style windows (simulation by default; BrainFlow synthetic board supported).
2. **Process** with 1–40 Hz band-pass and 50 Hz notch filtering, FFT band power, artifact flags.
3. **Decode** five commands (LEFT, RIGHT, FORWARD, BACKWARD, SELECT) with a scikit-learn classifier, with confidence and per-feature explanation.
4. **Act**: the Communicator speaks phrases and raises an emergency alert; virtual devices (robotic hand, wheelchair, cursor) show control applications (robotic hand, wheelchair, cursor) and an SSVEP communication board (YES, NO, HELP, WATER, FOOD, PAIN, CALL, THANKS).
5. **Close the loop**: low-confidence predictions are rejected, users verify or correct, and calibration retrains the model; only verified labels are ever used for training.
6. **Report**: measured latency, accuracy, ITR (bits/min), confusion matrices and a PDF session report.

## What is different
- Honest by design: every signal is labelled simulated; unverified predictions never become training labels; no fabricated latency or clinical claims.
- Explainable: the live decoder network shows real model weights and which features drove each prediction.
- Safety-first: confidence rejection, global E-STOP, photosensitivity-safe SSVEP (flicker off by default), no connection to real medical devices.
- Teachable: 22-page dashboard with animated brain, architecture and pipeline diagrams, plus a 19-channel spectrum/electrode/BCI-game lab.
- Engineering quality: FastAPI + WebSocket backend, SQLite persistence, 27 automated tests, Docker.

## Impact and 2036 vision
Near term: a free tool for students, labs and NGOs to explore BCI and assistive-tech ideas without expensive systems.
Medium term: validated on public EEG datasets and open EEG hardware; supervised pilots with clinicians after ethics approval.
By 2036: affordable personal BCIs may be realistic; this platform aims to be an open, trustworthy research and training tool on that path. (Vision, not a promise.)

## Tech stack
Frontend: HTML5, CSS3, vanilla JavaScript (Canvas and SVG animation). Backend: Python, FastAPI, WebSockets, SQLAlchemy + SQLite. Signal processing and ML: NumPy, SciPy (filters, FFT), scikit-learn (softmax regression, k-NN, cross-validation). Reporting: ReportLab (PDF). Hardware interface: BrainFlow (synthetic board tested). Data preparation script: MNE-Python. Testing: pytest (27 tests). Packaging: Docker.

## Status, limits and ethics (state these plainly)
- Signals are simulated. Accuracy figures on simulated data describe pattern classification, not decoding of human thoughts.
- Communicator usability test with N volunteers (see docs/USABILITY_TEST_PLAN.md): **<fill in: mean selection time, errors>**.
- Real clinical EEG: the signal-processing pipeline was validated on the public Bonn EEG dataset (500 real recordings, cross-validated by recording): seizure-vs-rest about 97% (sensitivity 94%, specificity 98%), 5-class about 77% per recording; see the Real EEG page and docs/bonn_results.md. Caveats: intracranial recordings, no patient IDs, easy binary task. This is not BCI decoding.
- Real **BCI** (motor-imagery) EEG: scripts/prepare_physionet.py and scripts/evaluate_real.py are ready; results: **<fill in after running, see README>**.
- Physical EEG hardware and any clinical use are untested and out of scope. Not a medical device.
- EEG is sensitive personal data: consent, on-device processing and security are design requirements.
- Add one line on how AI tools were used in development if the hackathon rules ask for it.

## Sources
- Down To Earth (Lancet Neurology 2021 stroke burden): https://www.downtoearth.org.in/amp/story/health/global-stroke-rates-climb-india-faces-51-spike-in-cases-over-30-years-lancet
- ThePrint (Lancet SEAR series, GBD 2019): https://theprint.in/health/india-recorded-13-lakh-stroke-cases-in-2019-highest-in-southeast-asia-finds-lancet-study/1798324
- WHO news release, Global Report on Assistive Technology: https://www.who.int/news-room/releases/16-05-2022-almost-one-billion-children-and-adults-with-disabilities-and-older-persons-in-need-of-assistive-technology-denied-access--according-to-new-report
