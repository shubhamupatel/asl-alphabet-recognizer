# ASL Alphabet Recognizer - Architecture & Decision Log

## Purpose

This is a from-scratch rebuild (v2) of my real-time sign language recognition project. The goal is to improve the performance of the project while also transitioning into more popular frameworks.

**v1 (summer 2025):** Built with OpenCV and TensorFlow. It recognized static ASL letters at about 90% accuracy, but only under clean lighting and a controlled background.

**v2 goal:** Keep accuracy while holding up under ordinary lighting and backgrounds.

## Scope

The system recognizes individual letters of the American Sign Language (ASL) fingerspelling alphabet, signed with one hand. Words, sentences, two-handed signs, facial grammar, and sign-to-text translation are out of scope.

## Design Principles

**Extensibility through clean seams, not up-front abstraction.** The system is a pipeline of independent stages. Each stage has a clear input and output, so one stage can be swapped or extended without rewriting the others. In practice:
- Landmark processing works on one frame at a time
- The list of class labels is defined in a single place
- Every classifier exposes the same interface: landmarks in → predicted letter and confidence out

Two known future extensions motivate this:
- Adding the motion letters J and Z, using a sequence model over landmarks from consecutive frames
- Upgrading to a hybrid approach, where a CNN classifies a cropped image of the hand

## Pipeline Overview

```
Webcam frame
  → hand detection + landmark extraction (MediaPipe: 21 points * x, y, z)
  → normalization (positions made relative to the wrist and scaled for hand size)
  → feature vector (63 numbers)
  → classifier
  → predicted letter + confidence
  → drawn onto the video frame
```

## Project Structure

```
asl-alphabet-recognizer/
├── asl/                  # building blocks: one module per pipeline stage
│   ├── __init__.py
│   ├── labels.py         # the letter labels (single source of truth)
│   ├── landmarks.py      # MediaPipe hand landmark extraction
│   ├── features.py       # normalization → 63-number feature vector
│   └── model.py          # classifier definition
├── prepare_kaggle_data.py  # Kaggle photos → landmark data (data/processed/kaggle.npz)
├── collect_data.py       # webcam → labeled landmark samples
├── train.py              # training loop + early stopping
├── evaluate.py           # final test-set evaluation
├── run_demo.py           # live webcam recognition
├── data/
│   ├── raw/              # downloaded datasets, never modified (not committed)
│   └── processed/        # landmark .npz files, one per source
└── models/               # hand_landmarker.task + trained weights
```

## Decision Log

Each entry records the options considered, what was chosen, and why.

### D0: Limit scope to the ASL alphabet

**Decision:** Classify single fingerspelled letters only

**Why:** It keeps the project small enough to understand end to end. It also turns the problem into standard multi-class classification over a small, fixed set of labels (at most 26), rather than open-ended sign-to-text translation.

### D1: Handle J and Z: static letters only for now

**Options considered:**
1. Recognize only the 24 static letters
2. Include J and Z as static snapshots
3. Include J and Z using a sequence model

**Decision:** Option 1. J and Z are logged as future work

**Why:**
- Every remaining letter is a single hand shape, so each prediction is one frame in → one label out
- Snapshot versions of J and Z would teach the model poses that aren't the real signs. A snapshot of J is indistinguishable from I
- A sequence model would roughly double the scope
- v1 also handled static letters only

**Extension path:** Because we use landmarks (D2), adding J and Z is an add-on rather than an overhaul. A motion sign becomes a short sequence of 63-number vectors (for example, 30 frames * 63). The extension needs three additions:
- Clip-based data collection
- A sequence model, such as an LSTM
- A rolling buffer of recent frames at inference time

The hard part is telling a held I from the start of a J. There are two options:
- One sequence model for all 26 letters
- The existing static classifier plus a motion check that hands off to a separate J/Z model

### D2: Recognition approach: hand landmarks + small classifier

**Options considered:**
1. A CNN trained directly on raw images of hands
2. A pretrained hand tracker (Google's MediaPipe) that extracts 21 hand landmarks, followed by a small classifier trained on those coordinates
3. A hybrid: MediaPipe locates and crops the hand, then a CNN classifies the crop

**Decision:** Option 2, landmarks plus a small classifier

**Why:**
- The classifier only ever sees hand geometry, so it is largely unaffected by background, lighting, and camera differences. Those differences are the main reason raw-image CNNs trained on public datasets fail on a real webcam (distribution shift)
- The model is small enough to train on a laptop CPU and to run in real time
- The dataset can be collected personally with a webcam
- Landmarks also make the J/Z extension cheap. A motion sign becomes a short sequence of 63-number vectors instead of a video clip of raw pixels

**Trade-offs accepted:**
- Hand detection relies on a pretrained model I didn't build
- Letters that differ mainly in thumb placement (A, S, T, M, N) may be harder to separate, and landmark depth (z) values are noisy
- If MediaPipe misses the hand in a frame, that frame gets no prediction
- The classifier is a simple network. The hybrid approach (option 3) is kept as a possible later upgrade

### D3: Framework: PyTorch

**Options considered:**
1. PyTorch
2. TensorFlow with Keras

**Decision:** PyTorch

**Why:**
- The training loop is written explicitly (forward pass, loss, backward pass, optimizer step), so every step of learning is visible and explainable. This directly serves the goal of the rebuild
- Owning the loop makes it easy to add validation checks and early stopping
- PyTorch is the most common framework in AI research and AI engineering roles
- Debugging works like ordinary Python

**Trade-offs accepted:**
- There is more boilerplate than Keras's one-line `model.fit()`
- v1 used TensorFlow, so the project is presented as v1 (TensorFlow, controlled conditions) → v2 (PyTorch + landmarks, built for ordinary conditions)

**Interview answer for "why switch frameworks?":** I wanted the training loop explicit so I understood every step, and PyTorch is what most of the field uses.

### D4: Run locally on a Windows laptop

**Options considered:**
1. My personal laptop
2. Google Colab

**Decision:** Local

**Why:**
- The webcam is needed for both data collection and the live demo. Colab can only reach a webcam through an awkward browser workaround
- The model has roughly 11k parameters, so a laptop CPU is able to train it comfortably. Colab's free GPU offers no real benefit here
- Everything lives in one project folder that can go on GitHub

**Trade-offs accepted:** I manage the installation and debug environment issues myself.

### D5: MediaPipe API: Tasks API (HandLandmarker)

**Options considered:**
1. The current Tasks API (`HandLandmarker`)
2. The legacy `mp.solutions.hands` API, by pinning an old MediaPipe version

**Decision:** Tasks API

**Why:**
- The legacy Solutions API was removed from MediaPipe starting with version 0.10.30, and support for it ended in March 2023
- The MediaPipe versions that support Python 3.13 (the version this project uses) don't include the legacy API at all

**Trade-offs accepted:**
- The Tasks API needs a separate downloaded model file (`hand_landmarker.task`)
- Its setup code is slightly more involved
- Most older tutorials, and v1, use the legacy API, so their code won't work as-is

### D6: Project structure: `asl` package + top-level scripts

**Options considered:**
1. Flat: all files in the top folder
2. A package (`asl/`) for reusable building blocks, with runnable scripts in the top folder
3. src layout: the package inside `src/`, installed with a `pyproject.toml` file

**Decision:** Option 2

**Why:**
- Each pipeline stage gets its own module, so the code mirrors the pipeline overview one-to-one
- Building blocks (imported) and scripts (run) are clearly separated
- Extensions slot in as new modules, such as a sequence model for J and Z
- No install step is needed: running a top-level script puts the top folder on Python's import path, so `import asl` works

**Trade-offs accepted:**
- Slightly more structure than a flat layout
- Scripts must stay in the top folder, since moving them into a subfolder would break imports from `asl`

### D7: MediaPipe running mode: VIDEO

**Options considered:**
1. IMAGE: every frame processed independently, as an unrelated photo
2. VIDEO: frames processed as a connected sequence, synchronously
3. LIVE_STREAM: asynchronous processing, with results delivered to a callback

**Decision:** VIDEO

**Why:**
- Every frame is processed, with no dropped frames, so each saved training sample reliably pairs one frame's landmarks with its label
- Tracking between frames skips full hand detection once the hand is found, making it faster and smoother than IMAGE
- Calls are synchronous, so the code reads top to bottom with no callbacks or threads

**Trade-offs accepted:**
- Every frame needs an increasing timestamp, which is handled inside `asl/landmarks.py`
- If processing is slow, the loop waits on it. LIVE_STREAM remains an option if the demo feels laggy, and the change would be contained within `asl/landmarks.py`

### D8: Landmark jitter: leave raw landmarks

**Context:** Testing showed accurate landmarks across lighting, angles, and hand shapes, with slight frame-to-frame jitter while the hand is held still. Causes include camera sensor noise (worse in dim light), lighting flicker, and natural hand movement.

**Options considered:**
1. Leave landmarks raw
2. Smooth landmarks over time, such as averaging each point with its recent positions
3. Leave landmarks raw, but smooth the predictions in the demo

**Decision:** Option 1, with option 3 held in reserve for the demo if predictions flicker

**Why:**
- Each training sample reflects exactly one frame, with no mixing between frames
- Jitter acts as natural variation in the training data, so the model learns to tolerate it
- Smoothing landmarks would contaminate training data: after switching letters during collection, smoothed values would blend the previous hand shape into samples saved under the new label
- Smoothing would make landmark extraction depend on past frames, breaking the "one frame at a time" design principle

**Trade-offs accepted:**
- Demo predictions may flicker between similar letters, such as M and N. If they do, prediction smoothing (option 3) addresses it without touching the training data

### D9: Normalization: wrist origin, palm-length scale

**Options considered:**
1. Wrist as the origin, divide by palm length (wrist to middle-finger knuckle, landmarks 0 → 9)
2. Wrist as the origin, divide by the farthest landmark's distance from the wrist
3. Center of all 21 points as the origin, divide by the hand's bounding-box size
4. MediaPipe's world landmarks (3D positions in meters, centered on the hand)

**Decision:** Option 1

**Why:**
- Subtracting the wrist removes where the hand is in the frame. Dividing by palm length removes how close it is to the camera
- The palm is rigid, so its length doesn't change between letters. A normalization reference must not depend on the thing being classified
- Options 2 and 3 fail that test: the farthest point, the center, and the bounding box all shift as fingers fold, so normalizing by them would partly erase the shape differences between letters
- World landmarks are 3D estimates that inherit z's noise, and would still need a size step to handle different hand sizes

**Trade-offs accepted:**
- Tilting the palm toward or away from the camera makes it look shorter, which slightly changes the scale
- x and y are fractions of the frame's width and height, so hand shapes are slightly stretched in the numbers. This is consistent as long as data collection and the demo use the same webcam, but would matter with a camera whose frame has a different shape. Resolved in D13, once D12 mixed square dataset images with webcam frames

### D10: Rotation: not normalized

**Options considered:**
1. Leave rotation alone
2. Normalize rotation, by rotating every hand so the wrist-to-middle-knuckle direction points straight up
3. Correct only small tilts, below some threshold angle

**Decision:** Option 1

**Why:**
- In ASL, hand orientation is part of what makes a letter. Several pairs share a handshape and differ mainly in orientation: H and U, G and Q, K and P
- Rotation normalization would make each of those pairs identical or nearly so, and no classifier could tell them apart
- Correcting only small tilts needs an arbitrary threshold. Letters near it would be corrected inconsistently, and deciding which tilts are accidental would require already knowing the letter

**Trade-offs accepted:**
- Natural, unintended tilt stays in the data, so the model has to learn to tolerate it. Training data should include natural variation in tilt

### D11: Left hands: mirror to right-hand form

**Options considered:**
1. Support only the hand used for data collection
2. Collect data with both hands
3. Mirror left hands so they match right hands, by flipping the sign of x after centering on the wrist

**Decision:** Option 3

**Why:**
- Left-handed fingerspelling is the mirror image of right-handed fingerspelling, so a model trained on only one hand misreads the other
- Mirroring gives every hand one standard form, so each letter is a single shape to learn
- Data collected with either hand serves both, without doubling collection
- Mirroring is one operation (x → −x after centering) and doesn't change palm length, so D9's scaling is unaffected

**Trade-offs accepted:**
- Correctness depends on MediaPipe's handedness label. A misidentified hand gets flipped wrongly and is likely misclassified
- `extract()` returns the handedness along with the landmarks
- MediaPipe's left/right labeling convention had to be verified. Live testing on our non-mirrored webcam frames confirmed the labels match the signer's real hands

**Revised by D19:** Which hands get mirrored is now decided by the signer's declared dominant hand, not MediaPipe's per-frame label. The mirroring itself is unchanged

### D12: Training data: public dataset + own recordings

**Options considered:**
1. Record my own webcam data only
2. Use a public dataset only (the Kaggle ASL Alphabet dataset: 87,000 images at 200×200 pixels, 29 classes)
3. Both: the public dataset for volume, plus my own webcam recordings, with the test set drawn from my own recordings

**Decision:** Option 3

**Why:**
- The public dataset provides tens of thousands of samples without recording time, plus a second signer's hands
- My own recordings match the demo's real conditions: my webcam, its frame shape, and VIDEO mode
- Drawing the test set from my own recordings means the final score measures real-world performance, rather than performance on public images that may be near-duplicates of training images

**Trade-offs accepted:**
- The Kaggle images were captured under controlled conditions and reportedly from a single signer, so they add volume more than variety
- The dataset's square images and my webcam's frame shape stretch landmark values differently, so x and y must be converted to consistent units (revisits D9's trade-off)
- Converting a folder of unrelated photos needs MediaPipe's IMAGE mode, so `asl/landmarks.py` must support it alongside VIDEO mode
- The dataset (about 1 GB) stays out of GitHub, and its license must be checked before use

### D13: Landmark units: pixels

**Context:** MediaPipe reports x as a fraction of the image's width and y as a fraction of its height. Once D12 mixed square 200×200 dataset images with a non-square webcam frame, the same hand shape would produce differently proportioned numbers from each source.

**Options considered:**
1. Convert to pixels inside `extract()`: x × width, y × height, z × width
2. Pass the frame's width and height into `features.py` and correct there
3. Switch to MediaPipe's world landmarks (3D positions in meters)

**Decision:** Option 1

**Why:**
- All three axes end up in the same unit regardless of camera. MediaPipe's z uses roughly the same scale as x, so it's multiplied by the width too
- Knowledge of the frame size stays in `asl/landmarks.py`, the only module that has the frame
- D9's normalization needs no changes, because dividing by palm length cancels out the starting unit
- World landmarks would bring back the reasons they were passed over in D9

**Trade-offs accepted:**
- `extract()` now returns pixel coordinates instead of 0–1 fractions, so anything that used the old format had to be updated

### D14: Depth values: keep z

**Options considered:**
1. Keep z (63 features)
2. Drop z (42 features)
3. Train both ways and compare (an ablation study)

**Decision:** Option 1, with option 3 planned once training works

**Why:**
- Several letters are distinguished mainly by depth: whether the thumb sits in front of, beside, or behind the fingers (A, S, E; M, N, T) and how far fingers curl toward the camera (C, O). These overlap with the letters D2 flagged as hardest to separate
- A neural network learns how much to rely on each input, so if z is mostly noise, the model can learn to give it less weight
- Keeping z leaves the choice to the model. Dropping it would remove information the model can never get back

**Trade-offs accepted:**
- z is the noisiest of the three values, which may make learning harder with limited data

**Follow-up:** Once training works, train with and without z and compare validation accuracy, especially on A, S, E, M, N, and T. The comparison uses validation data only, so the test set stays untouched

### D15: Wrist values: keep all 63 features

**Context:** After normalization the wrist is always the origin, so its three values are always zero.

**Options considered:**
1. Keep them (63 features)
2. Drop them (60 features)

**Decision:** Option 1

**Why:**
- An always-zero input is harmless. It contributes nothing to any neuron's output, and its weights receive zero gradient, so they never change
- Features line up exactly with MediaPipe's landmark numbering: feature 3·i + k is landmark i's coordinate k, and the vector reshapes cleanly back to (21, 3)
- Dropping them would shift every feature position by one landmark, an easy off-by-one error, for no real benefit

**Trade-offs accepted:**
- If features are ever standardized (subtract each feature's mean, divide by its standard deviation), the zero columns have a standard deviation of 0 and need special handling to avoid dividing by zero
- The weights attached to these inputs never train, so they slightly inflate the parameter count

### D16: Photo support: IMAGE mode via a constructor option

**Context:** D12 adds the Kaggle dataset's photos as a data source. Consecutive photos are unrelated, so VIDEO mode's tracking would carry one photo's hand position into the next.

**Options considered:**
1. One class with a `video_mode` option: VIDEO mode for the webcam, IMAGE mode for photos
2. Two separate classes, one per mode, sharing a parent class
3. VIDEO mode for everything, with made-up timestamps

**Decision:** Option 1

**Why:**
- All shared logic (color conversion, pixel units, handedness) exists once, so photos and webcam frames can never be processed differently. D12 depends on both sources producing identical features
- The modes differ only in the detection call, so one `if` is enough. Two classes would add structure for a one-line difference
- The option is a plain True/False, so callers never need to import anything from MediaPipe
- Tracking across unrelated photos would cause missed or misplaced detections

**Trade-offs accepted:**
- The class carries timestamp state that IMAGE mode never uses

**Note:** It doesn't matter whether the dataset's photos were mirrored like a selfie camera. A mirrored right hand looks like a left hand, so MediaPipe labels it "Left," and D11 mirrors it back to right-hand form

### D17: Saved data: raw landmarks, not features

**Options considered:**
1. Raw landmarks (the (21, 3) pixel array), plus handedness, letter, and source. Features are computed when the data is loaded
2. The final 63-number features, plus the letter
3. Both

**Decision:** Option 1

**Why:**
- Changing the feature processing, such as for D14's z ablation, only means recomputing features from the saved landmarks, without rerunning MediaPipe over tens of thousands of photos
- `landmarks_to_features` stays the single path to features for every source: the dataset, my recordings, and the live demo
- Recording each sample's source makes it possible to keep my recordings for the test set (D12), and to keep one recording session from being split across training and test data
- Handedness and original positions stay available for inspecting odd samples

**Trade-offs accepted:**
- Features are recomputed on every training run, which takes about a second

### D18: Saved data format: NumPy .npz, one file per source

**Options considered:**
1. NumPy `.npz`: one compressed file holding several named arrays
2. CSV: one row per sample
3. Pickle (Python's built-in object serialization)
4. Parquet (a typed, compressed table format)

**Decision:** Option 1, with one file per source

**Why:**
- Arrays load back with their exact types and shapes (float32, (N, 21, 3)), so nothing needs parsing or reshaping
- Binary storage is compact and fast, and NumPy is already a dependency
- One file per source (the Kaggle dataset, each recording session) works around .npz files not supporting appends, and keeps sources separate for splitting
- Pickle can run arbitrary code when a file is loaded and breaks across Python versions. Parquet would add two new dependencies

**Trade-offs accepted:**
- Files aren't human-readable, so inspecting one takes a line of Python

**Note:** Letters are stored as text ("K"), not as class indices. Indices shift if the label set changes (adding J would move K from 9 to 10), so they're computed at load time with `LETTER_TO_INDEX`

### D19: Handedness: declared per signer, not detected per frame

**Context:** After converting the Kaggle dataset, 4.9% of samples were labeled as left hands, even though the dataset reportedly comes from a single right-handed signer. The left labels clustered in letters where handedness is hard to see: M (25%), C (20%), and O (14%), versus under 1% for open, palm-forward letters like L, Y, and B. That pattern points to MediaPipe misjudging handedness, which under D11 would wrongly mirror those samples, and could do the same in the live demo.

**Options considered:**
1. Keep trusting MediaPipe's per-frame label (D11 as originally designed)
2. Use a declared dominant hand: Kaggle data is treated as right-handed, my recordings use the hand I record with, and the demo takes a setting
3. Mirror augmentation: train on every sample plus its mirror image, and ignore handedness entirely

**Decision:** Option 2

**Why:**
- A signer's dominant hand doesn't change from frame to frame, so it should be stated once rather than guessed every frame. This removes the error source entirely
- Every hand still reaches the classifier in one standard right-hand form, keeping D11's main benefit
- Mirror augmentation would give up that standard form, making the model learn two mirror-image shapes per letter

**Trade-offs accepted:**
- The signer's hand must be declared, so the demo needs a setting
- The Kaggle signer is assumed to be right-handed throughout. About 95% of samples being labeled right-handed strongly supports this
- `extract()` still returns MediaPipe's per-frame handedness, but only for diagnostics. It is never used for features

### D20: Recording method: countdown, then automatic burst

**Options considered:**
1. One sample per keypress
2. Continuous capture while recording is toggled on
3. Continuous capture, keeping only every Nth frame
4. Press a letter's key, wait through a short countdown, then record automatically until a fixed number of frames with a detected hand are saved

**Decision:** Option 4

**Why:**
- The countdown keeps transition frames (the hand still forming the sign) out of the data, avoiding mislabeled samples
- Every letter gets the same number of samples, working against the class imbalance seen in the Kaggle data
- Counting detected samples instead of seconds means letters MediaPipe sometimes loses, like M and N, still reach the full count
- The hand is free during the burst, so tilt and distance can be varied for natural variety (D10)

**Trade-offs accepted:**
- The script has to track three states: idle, counting down, and recording
- Consecutive frames are nearly identical, so samples from one burst aren't independent. Data splits must keep each recording session together (see D17)

## Open Decisions

None at the moment.