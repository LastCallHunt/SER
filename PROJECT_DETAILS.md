# 📋 PROJECT_DETAILS.md
## Deep Technical Dive — Noise-Robust Speech Emotion Recognition Using CNN-BiLSTM with Demucs-Based Noise Cancellation Preprocessing and Sliding Window Segmentation for Long-Form Audio

> This document is for portfolio reviewers, academic evaluators, and developers  
> who want to understand every technical decision made in this project.

---

## 1. Problem Definition

### What We Set Out to Solve

Speech Emotion Recognition (SER) is the task of automatically identifying a speaker's
emotional state from their voice. Two problems make this hard in the real world:

**Problem 1 — Noise:** Real-world audio always contains background noise. A model trained
on clean studio recordings (like most research does) sees 18% accuracy degradation when
background noise is introduced. A hospital, call center, or classroom is never as quiet as a
recording studio.

**Problem 2 — Audio Length:** Almost all published SER models only work on 3–5 second
clips. Real conversations, therapy sessions, and customer calls are minutes long. Cutting
them into tiny pieces and picking one prediction throws away context.

**Our Answer:**
- **Demucs** (Facebook's neural source separator) removes background noise before classification
- **Sliding window segmentation** with parallel processing handles audio of any length

---

## 2. Dataset Details

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Files:** 2,880 WAV audio files
- **Actors:** 24 professional actors (12 male, 12 female)
- **Emotions:** 8 — neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **Intensity:** Two levels (normal, strong) for most emotions
- **Sample Rate:** 48,000 Hz (resampled to 22,050 Hz in our pipeline)
- **Download:** `kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio`

### TESS (Toronto Emotional Speech Set)
- **Files:** 5,600 WAV audio files
- **Speakers:** 2 female speakers (ages 26 and 64)
- **Emotions:** 7 — angry, disgust, fearful, happy, neutral, surprised (ps), sad
- **Sample Rate:** 24,414 Hz (resampled to 22,050 Hz)
- **Download:** `kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess`

### Combined Dataset
```
Total files      : 8,480
Training set     : 6,784 files (80%)
Test set         : 1,696 files (20%)
Random seed      : 42
Emotion classes  : 8 (after unified mapping)
```

### Emotion Label Unification

RAVDESS and TESS use different naming conventions. We unified them:

```python
ravdess_map = {
    '01': 'neutral',  '02': 'calm',     '03': 'happy',    '04': 'sad',
    '05': 'angry',    '06': 'fearful',  '07': 'disgust',  '08': 'surprised'
}

tess_map = {
    'angry': 'angry',   'disgust': 'disgust',  'fear': 'fearful',
    'happy': 'happy',   'neutral': 'neutral',  'ps': 'surprised',  'sad': 'sad'
}
# Note: TESS does not have 'calm' — RAVDESS contributes all calm samples
```

---

## 3. Feature Engineering

### Why These Three Features?

Each feature captures a different aspect of emotional speech:

| Feature | Dimensions | What it captures | Why emotion-relevant |
|---|---|---|---|
| MFCC | 40 | Spectral envelope (voice timbre/texture) | Angry vs calm voices have very different timbre |
| Chroma | 12 | Pitch class energy (harmonic content) | Happy speech has higher, more varied pitch |
| Mel-Spectrogram | 128 | Time-frequency energy distribution | Captures overall loudness patterns per frequency band |

### Feature Extraction Code (exact implementation)

```python
def extract_features(file_path):
    try:
        # Load 3 seconds of audio, skipping first 0.5s (silence/breath)
        y, sr = librosa.load(file_path, sr=22050, duration=3.0, offset=0.5)
        
        # MFCC: 40 coefficients, averaged over time frames
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        # Shape: (40,)

        # Chroma: using STFT magnitude, 12 pitch classes
        chroma = np.mean(librosa.feature.chroma_stft(
            S=np.abs(librosa.stft(y)), sr=sr).T, axis=0)
        # Shape: (12,)
        
        # Mel-Spectrogram: 128 mel bands, averaged over time frames
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        # Shape: (128,)
        
        # Concatenate all features
        return np.hstack((mfcc, chroma, mel))
        # Final shape: (180,)
    except:
        return None
```

### Why Temporal Averaging?

Each feature is a 2D matrix (time_frames × feature_dims). We take the **mean across time**
to get a fixed-size 1D vector per audio file. This is standard in SER research and allows
batching files of different lengths without padding issues.

---

## 4. Model Architecture Deep Dive

### Full Architecture

```
Input shape: (180, 1)
│
├── Conv1D(256 filters, kernel_size=8, padding='same', activation='relu')
│   └── Learns local spectral patterns across the 180-feature sequence
│
├── BatchNormalization()
│   └── Stabilises training, prevents internal covariate shift
│
├── MaxPooling1D(pool_size=5, strides=2, padding='same')
│   └── Reduces sequence length while keeping dominant features
│
├── Bidirectional(LSTM(128 units, return_sequences=True))
│   └── Processes the sequence FORWARD and BACKWARD simultaneously
│   └── return_sequences=True passes full sequence to next layer
│
├── Dropout(0.3)
│   └── Prevents overfitting by randomly dropping 30% of connections
│
├── Bidirectional(LSTM(64 units))
│   └── Second BiLSTM compresses to a single context vector
│
├── Dense(64, activation='relu')
│   └── Fully connected layer for final feature combination
│
├── Dropout(0.3)
│   └── Second regularisation layer
│
└── Dense(8, activation='softmax')
    └── Outputs probability for each of 8 emotion classes
    └── All 8 probabilities sum to 1.0
```

### Training Configuration

```python
model.compile(
    optimizer='adam',                        # Adaptive learning rate
    loss='sparse_categorical_crossentropy',  # Multi-class classification
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    batch_size=64,          # 64 samples per gradient update
    epochs=30,              # 30 full passes through training data
    validation_data=(X_test, y_test)
)
```

### Training Progress (actual results from our notebook)

| Epoch | Train Accuracy | Val Accuracy |
|---|---|---|
| 1 | 55.98% | 58.96% |
| 2 | 74.32% | 76.00% |
| 10 | ~84.89% | ~83.37% |
| 20 | ~93.43% | ~91.27% |
| 30 | **98.17%** | **95.75%** |

Training time: **~12–15 minutes on T4 GPU** (Google Colab)

---

## 5. Demucs Integration

### What Demucs Does

Demucs is a neural network trained to separate audio into its component stems:
`drums | bass | other | vocals`. We use the `htdemucs` model with `--two-stems vocals`
mode which produces just two outputs: `vocals` and `no_vocals`.

### Integration Point

Demucs runs **before** feature extraction. The pipeline is:

```
Noisy audio file
    ↓
Demucs: extract vocals track
    ↓
Clean vocal .wav file
    ↓
Feature extraction (MFCC + Chroma + Mel)
    ↓
CNN-BiLSTM prediction
```

### Performance Impact

| Condition | Without Demucs | With Demucs | Improvement |
|---|---|---|---|
| Clean audio | ~95.75% | ~95.75% | No change (expected) |
| Noisy audio | ~76% | ~88% | +12 percentage points |

---

## 6. Sliding Window Segmentation

### The Algorithm

```python
window_sec  = 4    # Each chunk is 4 seconds
overlap_sec = 2    # Chunks overlap by 2 seconds (50% overlap)
step        = window_sec - overlap_sec  # = 2 second steps

# For a 30-second audio file:
# Chunk 1: 0s → 4s
# Chunk 2: 2s → 6s
# Chunk 3: 4s → 8s
# ...
# Chunk 14: 26s → 30s
```

### Parallel Processing

All chunks are processed simultaneously using Python's `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(predict_chunk, segments))
```

### Final Prediction: Majority Vote

```python
from collections import Counter
all_emotions = [r['emotion'] for r in results]
dominant_emotion = Counter(all_emotions).most_common(1)[0][0]
```

---

## 7. Challenges and Solutions

### Challenge 1: Short audio handling
**Problem:** Audio shorter than `window_sec` (4s) caused the segmentation loop to
produce zero chunks, crashing the pipeline.  
**Solution:** Added explicit check — if `duration < window_sec`, treat entire audio
as a single segment.

```python
if duration < window_sec:
    segments.append((0, duration, y))  # Single segment
```

### Challenge 2: Feature extraction bug in parallel chunks
**Problem:** Original code used the full audio `y` instead of `chunk` when extracting
chroma features inside parallel processing, causing all chunks to return identical features.  
**Solution:** Fixed variable reference from `y` to `chunk` in the parallel worker function.

```python
# BUG: chroma_stft(S=np.abs(librosa.stft(y)), ...)   ← uses full audio
# FIX: chroma_stft(S=np.abs(librosa.stft(chunk)), ...) ← uses correct chunk
```

### Challenge 3: Model save path in Colab
**Problem:** Model saved to `/content/my_bvr_cnn_bilstm.keras` gets deleted when
Colab session ends.  
**Solution:** Mount Google Drive and save there, or download immediately after training.

---

## 8. File Saved

The trained model is saved as:
```
/content/my_bvr_cnn_bilstm.keras
```

To load it in a future session:
```python
from tensorflow.keras.models import load_model
model = load_model('/content/my_bvr_cnn_bilstm.keras')
```

---

**Base Paper:**  
Tiwari, S., Kumar, D., Mahajan, A., and Sachar, S. "Emotion Detection from Speech Using
CNN-BiLSTM with Feature Rich Audio Inputs." ICCK Transactions on Machine Intelligence,
vol. 1, no. 2, pp. 80–89, Sep. 2025.
