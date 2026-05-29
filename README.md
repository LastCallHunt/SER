# 🎙️ Noise-Robust Speech Emotion Recognition Using CNN-BiLSTM with Demucs Preprocessing and Sliding Window Segmentation

> Detect human emotions from speech — even in noisy real-world conditions.  
> Built with **CNN-BiLSTM**, **Facebook Demucs Noise Cancellation**, and **Sliding Window Segmentation for Long-Form Audio**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Google Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-yellow.svg)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Val%20Accuracy-95.75%25-brightgreen.svg)]()
[![Demucs](https://img.shields.io/badge/Noise%20Cancellation-Demucs%20htdemucs-purple.svg)](https://github.com/facebookresearch/demucs)
[![Audio](https://img.shields.io/badge/Long%20Audio-Up%20to%2025%20mins-blue.svg)]()

---

## 📌 What This Project Does

Most speech emotion recognition (SER) systems fail in two ways:
1. **They break under noise** — accuracy drops up to 18% with background noise
2. **They only handle short clips** — typically 3–5 seconds maximum

This project solves both problems through a three-phase pipeline:

```
Raw Audio (any length) 
    → Demucs Noise Removal 
    → MFCC + Chroma + Mel Feature Extraction 
    → CNN-BiLSTM Classifier 
    → Emotion Label + Confidence Score
```

---

## 🏆 Results

| Metric | Value |
|---|---|
| **Validation Accuracy** | **95.75%** |
| Training Accuracy | 98.17% |
| Dataset Size | 8,480 audio files |
| Emotions Classified | 8 |
| Max Audio Length Supported | 25 minutes |
| Feature Vector Size | 180 dimensions |

### Per-Emotion Performance (after 30 epochs)

| Emotion | Notes |
|---|---|
| angry | Highest confidence — strong spectral features |
| happy | Very reliably detected |
| sad | Clear MFCC separation |
| fearful | Occasionally confused with surprised |
| surprised | Occasionally confused with fearful |
| neutral | Solid performance |
| calm | Distinguishable from neutral via pitch features |
| disgust | Well-separated via energy features |

---

## 📁 Project Structure

```
Speech-Emotion-Recognition-CNN-BiLSTM-Demucs/
│
├── semifinalproject.ipynb     # Main Google Colab notebook
├── README.md                  # This file
├── requirements.txt           # All dependencies
├── PROJECT_DETAILS.md         # Deep technical dive
├── setup_instructions.md      # How to run step by step
├── .gitignore                 # Git ignore rules
└── LICENSE                    # MIT License
```

---

## 🗃️ Datasets Used

| Dataset | Files | Emotions | License |
|---|---|---|---|
| **RAVDESS** | 2,880 WAV files | 8 emotions, 24 actors | CC-BY-NC-SA-4.0 |
| **TESS** | 5,600 WAV files | 7 emotions, 2 speakers | CC BY-NC-ND 4.0 |
| **Combined** | **8,480 files** | **8 unified emotions** | — |

**Emotion mapping:**

```
RAVDESS codes → neutral, calm, happy, sad, angry, fearful, disgust, surprised
TESS labels   → angry, disgust, fearful, happy, neutral, surprised, sad
```

**Train/Test Split:** 80% training (6,784 files) / 20% test (1,696 files)

---

## 🧠 Model Architecture

```
Input: (180, 1)  ← 180-dim feature vector reshaped for Conv1D
    │
    ▼
Conv1D(256 filters, kernel=8, padding='same', activation='relu')
BatchNormalization()
MaxPooling1D(pool_size=5, strides=2, padding='same')
    │
    ▼
Bidirectional LSTM(128 units, return_sequences=True)
Dropout(0.3)
Bidirectional LSTM(64 units)
    │
    ▼
Dense(64, activation='relu')
Dropout(0.3)
Dense(8, activation='softmax')   ← 8 emotion classes
```

**Why CNN + BiLSTM?**
- **CNN layers** extract local spectral patterns from the feature map
- **BiLSTM layers** capture temporal dependencies in BOTH forward and backward directions — critical for emotion which unfolds over time

---

## 🔊 Feature Extraction

Three complementary features are extracted from each audio clip using `librosa`:

```python
# Audio loaded at 22050 Hz, 3 seconds, 0.5s offset
y, sr = librosa.load(file_path, sr=22050, duration=3.0, offset=0.5)

mfcc   = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)   # shape: (40,)
chroma = np.mean(librosa.feature.chroma_stft(...).T, axis=0)                # shape: (12,)
mel    = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)     # shape: (128,)

features = np.hstack([mfcc, chroma, mel])  # Final shape: (180,)
```

| Feature | Dimensions | What it captures |
|---|---|---|
| MFCC | 40 | Spectral envelope / voice timbre |
| Chroma | 12 | Pitch class / harmonic content |
| Mel-Spectrogram | 128 | Time-frequency energy distribution |
| **Combined** | **180** | **Complete acoustic-emotional profile** |

---

## 🔇 Demucs Noise Removal

Facebook's [Demucs](https://github.com/facebookresearch/demucs) `htdemucs` model separates vocals from background noise before feature extraction:

```python
cmd = "python3 -m demucs.separate -n htdemucs --two-stems vocals 'input.wav' -o /content/cleaned_audio"
# Output: /content/cleaned_audio/htdemucs/{filename}/vocals.wav
```

**Why this matters:**
- Without Demucs: ~18% accuracy drop under noise (documented in literature)
- With Demucs: Maintains above 88% accuracy on noisy audio
- Recovery: ~12 percentage points restored

---

## ⏱️ Long Audio Sliding Window

For audio longer than 4 seconds, a parallel sliding window processes overlapping chunks:

```python
window_sec = 4      # 4-second chunks
overlap_sec = 2     # 50% overlap
step = window_sec - overlap_sec  # 2-second steps

# Parallel processing with ThreadPoolExecutor(max_workers=4)
# Final prediction: majority vote across all segments
```

**Processing capacity:** Up to 25 minutes of audio on T4 GPU

---

## 🚀 Quick Start (Google Colab)

### Step 1 — Open the notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

Upload `semifinalproject.ipynb` to Google Colab.

### Step 2 — Enable GPU
`Runtime → Change runtime type → T4 GPU`

### Step 3 — Upload Kaggle API key
The first cell asks you to upload `kaggle.json`.  
Get yours from: [kaggle.com → Settings → API → Create New Token](https://www.kaggle.com/settings)

### Step 4 — Run all cells
`Runtime → Run all`  
⏳ Feature extraction takes ~8–10 minutes. Training takes ~12–15 minutes on T4 GPU.

### Step 5 — Test your own audio
The demo cell (Cell 9) asks you to upload any `.wav` or `.mp3` file and returns:
- Predicted emotion
- Per-segment emotion timeline (for long audio)
- Before/after Demucs comparison

---

## 💾 Loading the Saved Model

After training, the model is saved as `my_bvr_cnn_bilstm.keras`:

```python
from tensorflow.keras.models import load_model
import pickle

model = load_model('/content/my_bvr_cnn_bilstm.keras')
print("Model loaded successfully")
```

---

## ⚠️ Limitations

- English speech only (RAVDESS + TESS are English datasets)
- Trained on acted speech — may differ from spontaneous natural speech
- Demucs takes ~30–60 seconds per minute of audio on T4 GPU
- No real-time streaming support (file-based only)
- Free Colab tier: ~4–6 hours GPU per day

---

## 🔮 Future Work

- [ ] Add attention mechanism over BiLSTM layers
- [ ] Train on multilingual datasets (IEMOCAP, MSP-Podcast)
- [ ] Real-time microphone streaming
- [ ] Mobile app / PWA deployment
- [ ] Cross-lingual transfer learning with wav2vec 2.0

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) — Ryerson University
- [TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) — University of Toronto
- [Facebook Demucs](https://github.com/facebookresearch/demucs) — Meta AI Research
- Base paper: Tiwari et al. (2025) — CNN-BiLSTM with Feature Rich Audio Inputs
