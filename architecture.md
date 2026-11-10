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
│   ├── data.py           # saved .npz files → features and labels
│   └── model.py          # classifier definition, saving, loading, and running
├── prepare_kaggle_data.py  # Kaggle photos → landmark data (data/processed/kaggle.npz)
├── collect_data.py       # webcam → labeled landmark samples
├── train.py              # training loop + early stopping
├── evaluate.py           # final test-set evaluation
├── run_demo.py           # live webcam recognition
├── data/
│   ├── raw/              # downloaded datasets, never modified (not committed)
│   ├── processed/        # landmark files derived from raw/, regenerable (not committed)
│   └── recordings/       # my webcam recordings, one .npz per session (committed)
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
- The model has about 18,000 parameters (D24), so a laptop CPU is able to train it comfortably. Colab's free GPU offers no real benefit here
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

**Option 3 adopted in D39:** The demo averages predicted probabilities over recent frames. The landmarks themselves stay raw

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

**Follow-up resolved by D38:** Permutation importance on `val1` showed the model relies heavily on z, especially for the depth-dependent letters, so z stays and the full ablation was skipped

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

### D21: Recording storage: `data/recordings/`, committed

**Context:** Recording saves only landmarks, never webcam frames, so a lost recording can't be regenerated. Unlike `kaggle.npz`, which can be rebuilt from the raw photos, recordings are original data.

**Options considered:**
1. Save in `data/processed/`, alongside `kaggle.npz`
2. Save in a separate `data/recordings/` folder, committed to GitHub
3. Save in `data/recordings/` without committing

**Decision:** Option 2

**Why:**
- Original and derived data stay separate, so clearing out regenerable files can never delete recordings
- GitHub acts as a backup
- Anyone cloning the repo can retrain the model from the same data
- It's my own data with no license restrictions, the files are small (around 1–2 MB per session), and they contain only hand-joint coordinates: no images, face, or background

**Trade-offs accepted:**
- Git can't show meaningful diffs for binary files, though each session file is written once and never edited

### D22: Recording preview: mirror the display only

**Options considered:**
1. No mirroring
2. Mirror the frame before extracting landmarks
3. Mirror only the displayed image, after extracting landmarks

**Decision:** Option 3

**Why:**
- A mirrored preview works like a bathroom mirror, so signing while watching feels natural
- Landmarks always come from the original, unmirrored frame. Mirroring before extraction would give every recorded sample mirror-image geometry, contradicting the Kaggle data's right hands while still being declared right-handed
- Display transformations must never leak into the data

**Trade-offs accepted:**
- Drawing order matters: landmarks are drawn before mirroring so they stay on the hand, and text is drawn after so it reads normally

### D23: Recording plan: separate sessions for training, validation, and test

**Options considered:**
1. One session as the test set, with validation taken from Kaggle data
2. Three sessions: one each for training, validation, and test
3. Four or more sessions, with two or more for training

**Decision:** Option 2, plus an optional non-dominant-hand session used only as an extra test of left-hand mirroring (D11, D19)

**Why:**
- Every role gets data from the real demo setup (my webcam, my hand). Validation needs this as much as test, since early stopping and the z ablation (D14) are tuned against it
- Assigning whole sessions to roles makes it impossible for near-duplicate frames to leak between training and evaluation (D20)
- The model trains on my hand in addition to the Kaggle signer's

**Plan:**
- Kaggle data is used for training only
- In each session, every letter is recorded in two bursts of 100 samples (4,800 samples per session), repositioning between bursts, since most of a burst's variety comes from where it starts
- Within each burst: slowly vary tilt, distance, and position in the frame
- Between sessions: vary lighting, background, time of day, and clothing. The test session uses conditions the training session didn't have, while staying realistic for the demo
- Session names state their role: `train1`, `val1`, `test1`, and `test_left1`

**Trade-offs accepted:**
- Roughly 30 minutes of recording, ideally spread across different days
- My training data (about 4,800 samples) is small next to Kaggle's 55,368, which may need weighting during training. Resolved in D25

**Revised by D36:** A second training session, `train2`, was added after per-letter results on `val1` showed gaps in the training data's coverage of hand angles. The plan is now four sessions, plus the optional `test_left1`

### D24: Model architecture: two hidden layers with ReLU and dropout

**Options considered:**
1. One hidden layer: 63 → 128 → 24
2. Two hidden layers: 63 → 128 → 64 → 24, with dropout
3. Deeper and wider, such as three hidden layers of 256

**Decision:** Option 2, with ReLU activations and a 20% dropout rate

**Why:**
- Without a non-linear activation, stacked linear layers collapse into a single linear layer, so ReLU goes between layers. ReLU passes gradients through intact for positive inputs, avoiding the vanishing gradients of sigmoid and tanh
- Two hidden layers can build combinations of combinations, which subtle letters (M/N/T, A/S/E) may need. One layer risks underfitting
- Dropout counters the extra capacity's tendency to memorize, which matters because the Kaggle data contains many near-duplicate frames. Option 3 would be overkill for 63 inputs and invite memorization
- At about 18,000 parameters, it still trains in minutes on a laptop CPU

**Trade-offs accepted:**
- The hidden sizes (128, 64) and dropout rate are informed starting guesses. They're constructor arguments, so they can be tuned on validation data (never test data)
- The model outputs raw scores (logits). Softmax is applied only when a confidence is needed, since PyTorch's cross-entropy loss applies it internally

### D25: Training data imbalance: weighted sampling, with the natural mix as a baseline

**Context:** Kaggle provides 55,368 training samples and `train1` provides 4,800, so about 92% of every shuffled batch, and of every gradient step, comes from Kaggle. The model minimizes the average loss, so wherever the two sources disagree it favors Kaggle, the source less like the demo. Kaggle's letters are also uneven, since MediaPipe found fewer hands in some letters' photos (M and N especially), while my recordings have equal counts per letter (D20).

**Options considered:**
1. Train on the natural mix
2. Weighted sampling: draw training batches so every source and every letter comes up equally often
3. Loss weighting: see every sample once per epoch, but scale each sample's loss by how rare its group is
4. Undersample Kaggle to the size of my recordings
5. Pretrain on Kaggle, then fine-tune on my recordings

**Decision:** Option 2, with option 1 kept as a switchable baseline

**Why:**
- One mechanism fixes both imbalances: each sample is weighted by 1 / (size of its source-and-letter group), so every source and every letter is drawn equally often
- The loss function and training loop stay unchanged, since balancing happens entirely in which samples get drawn
- Loss weighting would put large weights on a few samples per batch, making gradient steps jumpier, and PyTorch's built-in class weights cover letters but not sources
- Undersampling throws away about 91% of the Kaggle data
- Fine-tuning adds a second training stage with its own hyperparameters, and risks overwriting what Kaggle taught (catastrophic forgetting)
- How much the imbalance actually hurts can only be measured, so the natural mix stays available and the two are compared on validation data (never test data)

**Trade-offs accepted:**
- My samples are drawn about 11.5 times as often as Kaggle's on average, so the model sees the same near-duplicate bursts repeatedly and could memorize them. Dropout (D24) and early stopping on `val1` guard against this
- Sampling is done with replacement, so an epoch is no longer one pass over every sample, and the number of draws per epoch becomes a setting of its own. Resolved in D29
- Equal weight per source is a starting point, not a measured optimum. The ratio between sources could be tuned on validation data later

**Note:** Only training data is rebalanced. `val1` comes from a single source with equal letter counts, so it is evaluated as-is

**Revised by D33:** Measured on `val1`, the natural mix beat weighted sampling in both comparisons (99.0% vs. 96.6% validation accuracy at each run's best epoch, after the recordings were corrected), so the natural mix is now the default. Weighted sampling remains available through the switch

### D26: Data loading: a shared `asl/data.py` module

**Context:** `train.py` and `evaluate.py` both turn saved `.npz` files into features and labels: training and validation data in one, test data in the other.

**Options considered:**
1. Loading code inside `train.py`, with `evaluate.py` keeping its own copy
2. A new `asl/data.py` module, imported by both scripts
3. `evaluate.py` importing the loading function from `train.py`

**Decision:** Option 2

**Why:**
- Training, validation, and test data all go through one loading path, so they can never be processed differently. This extends D17's single path to features to loading as a whole
- Separate copies could drift apart silently. For example, dropping z for the D14 ablation in only one copy would feed the test set different features from the ones the model trained on
- Reusable building blocks belong in `asl/` and runnable scripts in the top folder (D6). Importing from `train.py` would blur that split and make evaluation depend on the training script

**Trade-offs accepted:**
- One more module to maintain

**Note:** The live demo doesn't use this module. It gets landmarks straight from the webcam and calls `landmarks_to_features` directly

### D27: Data module output: NumPy arrays, with DataLoaders built in each script

**Context:** D26 made `asl/data.py` the shared loader for `train.py` and `evaluate.py`, but the two scripts need their data batched differently. Training draws from two files through a weighted sampler or plain shuffling (D25), while validation and test each read one file in plain batches.

**Options considered:**
1. Return ready-made training and validation DataLoaders
2. Return NumPy arrays of features, labels, and sources, and let each script build its own tensors and DataLoaders
3. A custom PyTorch `Dataset` class

**Decision:** Option 2

**Why:**
- The module has one job, turning files into arrays, so it changes only when the saved format changes, not whenever training is tuned (the single-responsibility principle)
- Sampling, batch size, and D25's switch are training settings, so they belong in `train.py`. `evaluate.py` needs plain batches of `test1` alone, which a ready-made training and validation pair can't provide
- Roles stay with the scripts that assign them (D23). The module loads whatever files it's given and never decides which are training or validation
- A custom `Dataset` mainly helps with data too large to hold in memory. All the features fit in about 15 MB, and PyTorch's built-in `TensorDataset` already handles in-memory data
- Like `features.py`, the module needs only NumPy, so saved data can be inspected without loading PyTorch

**Trade-offs accepted:**
- Each script repeats a few lines of tensor and DataLoader setup

**Note:** Sources are returned alongside features and labels because D25's sampler weights each sample by its source and letter. The declared hand is used up by the feature conversion (D19) and isn't returned

### D28: Batch size: 64

**Options considered:**
1. 32
2. 64
3. 256
4. The full training set as one batch

**Decision:** 64

**Why:**
- Mini-batches balance gradient accuracy against the number of updates. 64 is the most common starting point, at about 940 steps per epoch
- With a model this small, a batch of 32 spends most of each step on fixed overhead, so epochs run slowly on a CPU
- Larger batches give fewer updates per epoch and less of the gradient noise that acts as mild regularization. A full batch gives only one update per pass through the data
- With balanced sampling (D25), each batch holds about 32 samples from my recordings

**Trade-offs accepted:**
- 64 is a starting point, not a measured optimum. It's a constant at the top of `train.py`, so it can be tuned on validation data (never test data)

**Note:** Batch size matters only for training. Validation reuses the same value for convenience: nothing learns during validation, and no layer's output depends on the rest of its batch, so it changes speed but not results

### D29: Draws per epoch: the full training set's size

**Context:** D25's weighted sampler draws with replacement, so an epoch no longer means one pass over every sample. The number of draws per epoch sets how many gradient steps happen between validation checks.

**Options considered:**
1. The full training set's size (60,168 draws)
2. Twice my recordings' size (9,600 draws)
3. Twice Kaggle's size (110,736 draws)

**Decision:** Option 1

**Why:**
- A balanced epoch has the same number of steps as a natural-mix epoch (about 940, D28), so epoch counts, patience, and learning curves compare directly between D25's two modes
- Validation runs at the same pace in both modes
- Shorter epochs would make validating all of `val1` a large share of the run. Longer ones would draw each of my samples around 11 times between checks, giving the model time to memorize my bursts before validation notices

**Trade-offs accepted:**
- Each Kaggle sample is drawn about 0.5 times per epoch, so roughly 58% go undrawn in any single epoch. This evens out across epochs: after five, only about 7% have never been drawn
- Each of my samples is drawn about 6 times per epoch

**Note:** The count is the training set's actual size, measured at load time rather than hardcoded, so it adjusts automatically if more training sessions are added

### D30: Optimizer: Adam, learning rate 0.001

**Options considered:**
1. Plain SGD
2. SGD with momentum
3. Adam
4. AdamW (Adam with weight decay)

**Decision:** Adam, with a learning rate of 0.001

**Why:**
- Adam combines momentum, which smooths each step's direction across noisy mini-batches, with a per-weight step size based on how large that weight's gradients usually are. This makes it much less sensitive to the learning rate than SGD
- Its standard learning rate of 0.001 works well for small networks with little tuning, keeping attention on the comparisons that matter, such as D25's sampling modes and D14's z ablation
- Weights connected to inputs with different gradient scales, such as the noisier z values (D14), get steps scaled to suit them
- Plain SGD is very sensitive to the learning rate and zigzags on noisy gradients. Momentum helps, but still needs careful learning-rate tuning
- AdamW's weight decay would add another regularizer to tune, on top of dropout (D24) and early stopping

**Trade-offs accepted:**
- Adam's updates are harder to explain step by step than SGD's
- Carefully tuned SGD with momentum sometimes generalizes slightly better. That comparison could be run on validation data later
- The learning rate is a starting value, not a measured optimum. It's a constant at the top of `train.py`, so it can be tuned on validation data (never test data)

### D31: Early stopping: validation loss, patience of 5

**Options considered:**
1. Watch validation loss
2. Watch validation accuracy
3. Both: stop on validation loss, but keep the epoch with the best validation accuracy

**Decision:** Option 1, with a patience of 5 epochs and a limit of 100 epochs

**Why:**
- Validation loss is continuous and sensitive. It moves whenever the model's confidence changes, even when no prediction flips
- It's the quantity the model is trained to reduce, and it shows overfitting first, as the model becomes confidently wrong on some validation samples
- Accuracy moves in steps of 1/4,800, so epochs often tie, and it ignores confidence, which the demo will display and may smooth predictions with (D8)
- Watching both would let the two disagree about which epoch is best, making the logic harder to write and explain
- Any decrease in validation loss resets the patience counter, so a run of tiny improvements could keep training going for a long time. The 100-epoch limit is a safety net against that

**Trade-offs accepted:**
- Loss can rise slightly while accuracy still improves, so the kept epoch isn't guaranteed to have the highest validation accuracy. Accuracy is printed every epoch, so this stays visible
- Validation loss bounces between epochs, so a patience of 5 may stop on a temporary bump before the real minimum. The best snapshot is always kept, so raising the patience could only cost time, and it can be raised if runs seem to stop early
- Keeping the best of many epochs makes the best validation loss slightly optimistic. The test set (D23) provides the honest number

### D32: Saved model: a self-describing checkpoint at a fixed path

**Context:** `evaluate.py` and the demo load the trained model from a file, so the file must contain everything they need to rebuild the model and turn its outputs into letters.

**Options considered:**
1. The weights only
2. A dictionary of the weights, the model's settings, the letter list, and training information
3. The whole model object, saved with pickle

**Decision:** Option 2, saved to a fixed path, `models/letter_classifier.pt`, that every run overwrites

**Why:**
- The file is self-describing. Loaders rebuild the exact model from its saved settings, and map outputs to letters with its own letter list rather than whatever `LETTERS` happens to be at load time
- Saving the letter order with the weights prevents training/serving skew, where an output index means one letter in training and another in the demo. A loader can compare the saved list with the current `LETTERS` and fail loudly on a mismatch
- The training information (best epoch, validation loss and accuracy, sampling mode) records which run produced the file
- The file holds only tensors, numbers, strings, and lists, so it loads with PyTorch's safe `weights_only=True` mode
- A weights-only file says nothing about the settings it needs, so after tuning, an old file stops loading with no record of why
- A pickled model object stores a reference to the class rather than the class itself, so it breaks when `model.py` changes or moves. It also can't use safe loading, which brings back the code-execution risk D18 rejected pickle for

**Trade-offs accepted:**
- Every run overwrites the previous model, including experiment runs such as D25's baseline and D14's z ablation. Their printed validation numbers remain for comparison, but the run whose model should be kept has to go last. An `--output` argument can be added if keeping several models becomes useful
- A few more lines on both the saving and loading sides

### D33: Sampling default: the natural mix

**Context:** D25 chose weighted sampling, keeping the natural mix as a switchable baseline to compare on validation data. Both comparisons below used seed 0.

The first comparison used `train1` and `val1` sessions later found to contain incorrectly signed letters:
- Natural mix: best validation loss 0.459 at epoch 2, with 85.2% validation accuracy
- Weighted sampling: best validation loss 0.693 at epoch 1, with 77.3% validation accuracy

After both sessions were re-recorded with corrected signs, the comparison was rerun:
- Natural mix: best validation loss 0.046 at epoch 7, with 99.0% validation accuracy
- Weighted sampling: best validation loss 0.121 at epoch 3, with 96.6% validation accuracy

**Options considered:**
1. Make the natural mix the default, keeping weighted sampling as the switchable alternative
2. Tune a middle ground, such as weighting my recordings at 25% instead of 50%, or balancing letters but not sources
3. Keep weighted sampling as the default and expect more training sessions to fix it

**Decision:** Option 1

**Why:**
- The natural mix won both comparisons. In the rerun, it had higher validation accuracy at every epoch, and lower validation loss from epoch 3 onward
- The likely cause is the risk D25 accepted. Weighted batches drew each of my near-duplicate `train1` samples about 6 times per epoch, so the model learned details specific to that session that didn't carry over to `val1`. In both comparisons, its validation loss bottomed out sooner (epoch 3 vs. 7 in the rerun)
- In the first comparison, weighted sampling also amplified the incorrectly signed letters about 6 times as much as the natural mix did, which likely explains much of that larger gap. The rerun's gap is smaller but still clear
- A middle ground would mean more comparisons on `val1`, and every extra comparison makes its score more optimistic
- Keeping the worse setup would rest on a guess about future data

**Trade-offs accepted:**
- Kaggle's letters stay uneven, so M and N remain underrepresented in training, though nothing in the results shows this hurting
- About 92% of training samples come from Kaggle again
- Each comparison used one seed. The rerun's accuracy gap at every epoch makes luck unlikely, but a second seed would confirm it
- 99.0% is the best of 12 epochs, on a validation session now used for several comparisons, so it's optimistic. The test session (D23) provides the honest number

**Note:** Correcting the signs raised validation accuracy from 85.2% to 99.0% with no change to the model or training. Data quality, not generalizing across sessions, was the main limitation in the first comparison

**Follow-up:** Retest weighted sampling once more training sessions exist, since each of my samples would then be drawn less often. Retested after adding `train2` (D36): weighted sampling reached 97.0% validation accuracy, with V collapsing to 52%, against 98.5% for the natural mix, so the natural mix stays the default

### D34: Shared classifier code: saving, loading, and running the model in `asl/model.py`

**Context:** `evaluate.py` needs to run a model over a session the way `train.py`'s validation does, but with every individual prediction, for per-letter results and a confusion matrix. It and the live demo also both need to load D32's checkpoint into a ready model.

**Options considered:**
1. `evaluate.py` imports the validation code from `train.py`
2. `evaluate.py` keeps its own copies of the measuring loop and the checkpoint loading
3. Shared functions in `asl/model.py`: save a checkpoint, load it into a ready model, and run a model over a loader to return every sample's scores

**Decision:** Option 3

**Why:**
- One function runs the model for both validation and testing, so test results are measured by exactly the code that measured validation (training/serving consistency)
- D32's checkpoint format lives in one place, next to the class it rebuilds, so saving and loading can't disagree
- Returning every sample's 24 scores serves every caller. Validation computes loss and accuracy from them, evaluation adds per-letter results and a confusion matrix, and the demo's confidences come from the same scores
- Computing the loss over all scores at once gives the exact average per sample, replacing `validate()`'s size-weighted running totals
- Importing from `train.py` would repeat the backwards dependency D26 rejected, and copies would drift apart, with the checkpoint format spread across three files

**Trade-offs accepted:**
- `validate()` and the checkpoint saving in `train.py` were reworked soon after being written
- `model.py` grows from a class definition into the classifier's whole lifecycle: defining, saving, loading, and running it. This still matches D6's one module per pipeline stage

**Note:** Inputs are still processed in batches, and only the 24 scores per sample are kept. Memory stays small even with a much larger model, such as D2's hybrid upgrade

### D35: Evaluation report: accuracy per letter and the most common confusions

**Context:** A confusion matrix counts how often each true letter (a row) was predicted as each letter (a column). At around 99% accuracy, nearly every cell off the diagonal is zero, so the full 24 × 24 grid is hard to read.

**Options considered:**
1. Print the full matrix in the terminal
2. Print accuracy per letter, worst first, plus a list of the most common confusions
3. Save the matrix as a heatmap image

**Decision:** Option 2

**Why:**
- It answers the questions the report exists for: which letters fail, and what they get mistaken for
- The full matrix is over 100 characters wide, and its few real errors are hard to spot among the zeros
- Errors concentrated in specific letters, like the incorrectly signed recordings behind D33's first comparison, are invisible in overall accuracy
- A heatmap needs a new dependency (matplotlib) and a separate file to open after every run

**Trade-offs accepted:**
- The whole matrix is never shown at once. It's still computed, so the display can be extended later

**Follow-up:** Add a heatmap image for the README once the final test results exist

### D36: Add a second training session, `train2`

**Context:** On `val1`, 42 of the 47 errors came from two letter pairs: V and K (V → K: 22, K → V: 4) and O and C (O → C: 16). Every V and O error fell in that letter's second burst, in contiguous runs (O's were 16 frames in a row), while both letters' first bursts were 100% correct. Those bursts included realistic changes in distance and hand angle. This points to gaps in the training data's coverage of those angles, not to incorrectly signed samples or a flaw in the model.

**Options considered:**
1. Record a second training session with broad variety in angle and distance
2. Data augmentation: synthetically rotate and scale training samples
3. Run D14's z ablation first
4. Accept the current model and move on to the test session and the demo

**Decision:** Option 1, followed by D14's z ablation on the larger training set

**Why:**
- It addresses the likely cause, missing coverage, with real data. Fixing data produced the project's largest improvement so far (D33)
- Augmentation risks blurring letters that differ mainly by orientation (D10), such as K and P, while K is already confused with V. Rotated landmarks with noisy z values also may not resemble real camera angles
- The z ablation measures whether depth helps but adds no coverage, so it's more informative once the training set is larger
- Accepting the model would leave a weakness the demo would show visibly. Half-second runs of wrong letters are too long for prediction smoothing (D8) to hide
- With twice as much of my own data, each of my samples would be drawn about half as often under weighted sampling, so D33's follow-up can be retested

**Trade-offs accepted:**
- About 10 more minutes of recording, plus retraining
- `train2` should vary angle and distance broadly rather than recreate `val1`'s failing poses. Copying them would tune the training data to `val1` and make its score optimistic
- `val1` is used for yet another comparison, adding to its optimism. The test session (D23) remains the honest check

**Plan:**
- The same structure as the other sessions: two bursts of 100 samples for every letter (D20, D23)
- Conditions different from both `train1` and `val1`: lighting, background, clothing, or time of day
- A somewhat wider range of turns, tilts, and distances than `train1`, while staying realistic for the demo

**Result:** See D37

### D37: Keep `train2` after auditing it

**Context:** Adding `train2` (D36) changed the natural-mix model's `val1` accuracy from 99.0% to 98.5%. V improved from 89.0% to 96.5%, but new errors appeared: A → S (4 to 25), M → N (0 to 13), and P → Q (0 to 7). To check whether `train2` contained bad samples, a model trained without it (Kaggle and `train1`, reproduced exactly with seed 0) was run on `train2`. It scored 98.7%, and its main disagreements were the reverse of `val1`'s new errors: Q → P (40) and S → A (10). The Q disagreements formed one contiguous stretch of about 35 frames in the second burst. Their middle fingertip's distance from the wrist (1.33 palm lengths) was closer to correctly formed Q's (1.39) than to P's (1.50), so they appear to be correct Q's at an angle the older data didn't cover.

**Options considered:**
1. Keep `train2` and move on to D14's z ablation, then the test session
2. Drop `train2` and train on `train1` only
3. Re-record `train2`
4. Keep `train2`, and record `train3` later if the test session shows the same letter pairs failing

**Decision:** Option 1, with option 4 in reserve

**Why:**
- The audit found no sign of malformed samples, and `train2` fixed the V weakness, which the demo would have shown
- The new errors are best explained by one-sided coverage. `train2` added letters at new angles without their neighboring letters at those angles, which shifted the boundaries between them. More coverage on both sides fixes that, not less data
- The overall difference (98.5% vs. 99.0%) is within one run's epoch-to-epoch swings, which exceeded 3 points
- Choosing training data by whichever scores best on `val1` would fit the data to one session's quirks
- A re-recording would cover different angles by chance and could create new conflicts

**Trade-offs accepted:**
- A and S are noticeably weaker on `val1`, with A at 87.5%
- The audit's middle-finger check separates Q and P only weakly (by about 0.1 palm lengths), so it supports rather than proves that the Q stretch was valid. The 10 S disagreements weren't inspected individually
- M → N isn't explained by the audit, which only flags samples the older model got wrong

### D38: Depth values: keep z, confirmed by permutation importance

**Context:** D14 kept z and planned an ablation: train with and without z, then compare on validation data. A full ablation would touch five files (features, data loading, the model's input size, the checkpoint, and evaluation) and would need several seeds, since validation accuracy swings by over 3 points between epochs. A cheaper check, permutation importance, shuffles the z values among `val1`'s samples while leaving x, y, and the labels in place, then measures how much the current model's accuracy drops.

Overall accuracy fell from 98.5% to 88.2%. The largest drops were M (93.5% to 42.5%), K (96.0% to 59.0%), A (87.5% to 53.5%), O (93.5% to 68.5%), and T (100% to 82.0%), while N and C stayed at 100%. Of the 494 correct predictions lost, about 407 came from the ten depth-related letters checked, and the other 14 letters lost about 87 between them.

**Options considered:**
1. Keep z and close D14's follow-up with the permutation result
2. Run the full ablation anyway

**Decision:** Option 1

**Why:**
- The model relies heavily on z, and the losses are concentrated in the letters D14 expected to depend on depth: thumb placement (A, S, E, M, T) and finger curl (O). Nonsense inputs alone would have hurt every letter about equally
- Without depth, M collapses into N and O into C, pairs that differ mainly in depth, so a model trained without z is unlikely to make up for it
- No feature change is coming, so the test session (D23) can go ahead
- The full ablation would take a five-file change and several seeded runs to confirm what this already strongly suggests, plus another round of comparisons on `val1`

**Trade-offs accepted:**
- Permutation importance measures how much this model relies on z, not whether a model trained without z could do as well
- Pairing one sample's depth with another's x and y creates combinations that never occur naturally, so the drop likely overstates z's importance somewhat

### D39: Demo smoothing: average probabilities over the last 10 frames

**Context:** D8 kept landmarks raw and held prediction smoothing in reserve for the demo. Per-frame predictions can jump between similar letters, and the displayed letter needs to stay steady while still changing promptly when the signer moves to a new letter.

**Options considered:**
1. Require the same prediction N frames in a row before switching
2. Average the probabilities over the last N frames, and show the top letter of the average
3. Majority vote over the last N predictions
4. An exponentially weighted average of the probabilities

**Decision:** Option 2, with N = 10

**Why:**
- It uses confidence. A few weak wrong frames are outvoted by strong correct ones, without needing a perfect streak
- A streak rule resets on any single wrong frame. For a letter predicted correctly 80% of the time at some angle, 10 correct frames in a row happens only about 11% of the time, so the display could stay stuck on the previous letter
- The averaged probability also drives the confidence display and the "no letter detected" threshold, so one mechanism handles all three
- "The last 10 frames" is easier to explain and tune than an exponential decay rate
- Smoothing changes only what's displayed, never the training data (D8)

**Trade-offs accepted:**
- About a third of a second of delay before a new letter appears, at roughly 30 frames per second
- Runs of errors longer than the window still reach the screen, like `val1`'s 16-frame O → C run. Smoothing removes flicker but doesn't fix the V/K and O/C weaknesses
- N = 10 is a starting value, to be tuned by watching the demo

**Note:** The window is cleared whenever no hand is detected, so a letter from before the hand left can't carry over when it returns

### D40: Demo confidence threshold: 0.7

**Context:** When the model isn't confident about any letter, the demo shows "No letter detected" instead of a guess (a reject option). On `val1`, the model's per-frame confidence (its highest softmax probability) was usually lower when it was wrong: half of its 73 mistakes fell between 0.5 and 0.6. The measured trade-offs:
- 0.6: shows 98.2% of correct predictions, hides 49.3% of wrong ones
- 0.7: shows 96.5% of correct predictions, hides 69.9% of wrong ones
- 0.8: shows 93.6% of correct predictions, hides 82.2% of wrong ones
- 0.9: shows 90.6% of correct predictions, hides 90.4% of wrong ones

**Options considered:**
1. 0.6
2. 0.7
3. 0.8
4. 0.9

**Decision:** Option 2, applied to the 10-frame average (D39)

**Why:**
- It catches most mistakes while barely affecting correct letters. Beyond it, the gains shrink: 0.8 hides about twice as many correct frames (about 303 vs. 165) to catch about 9 more wrong ones
- Choosing from measured rates on validation data avoids relying on softmax confidences being calibrated
- Smoothing should hide fewer correct letters than the per-frame numbers suggest, since a steady letter's occasional weak frames are averaged up by the strong ones around them

**Trade-offs accepted:**
- About 22 of 4,800 wrong frames still pass at this threshold, including confident errors above 0.9 that no reasonable threshold catches
- The rates are per frame, while the demo thresholds the 10-frame average, so the demo's actual behavior will differ somewhat
- 0.7 is a starting value, to be tuned by watching the demo

## Open Decisions

None at the moment.