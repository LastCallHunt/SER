# 🛠️ Setup Instructions
## How to Run — Noise-Robust Speech Emotion Recognition Using CNN-BiLSTM with Demucs Preprocessing and Sliding Window Segmentation

> Everything runs in **Google Colab** — no local installation needed.  
> Estimated total time: **25–35 minutes** (mostly downloading and training)

---

## Prerequisites

Before you start, you need:

1. **A Google account** (for Google Colab)
2. **A Kaggle account** (free) — to download datasets
3. **Your Kaggle API key** (`kaggle.json`)

### How to get your Kaggle API key

1. Go to [kaggle.com](https://www.kaggle.com) and sign in
2. Click your profile picture → **Settings**
3. Scroll to **API** section
4. Click **Create New Token**
5. A file called `kaggle.json` will download to your computer
6. Keep this file — you'll need it in Step 3

---

## Step-by-Step Instructions

---

### Step 1 — Open the Notebook in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `semifinalproject.ipynb` from this repository
4. The notebook will open

---

### Step 2 — Enable GPU (Important — do this FIRST)

1. In Colab, click **Runtime** in the top menu
2. Click **Change runtime type**
3. Under **Hardware accelerator**, select **T4 GPU**
4. Click **Save**

> ⚠️ Without GPU, training will take 3–4 hours instead of 15 minutes.

---

### Step 3 — Run Cell 1: Environment Setup

Click the ▶ button on **Cell 1**.

This cell will:
- Install Demucs (~1–2 minutes)
- Ask you to **upload your `kaggle.json`** file

When the file upload dialog appears:
1. Click **Choose Files**
2. Select your `kaggle.json` file
3. Wait for it to upload

Expected output:
```
Environment ready.
```

---

### Step 4 — Run Cell 2: Download Datasets

Click ▶ on **Cell 2**.

This downloads:
- RAVDESS dataset (429 MB) — takes ~2–3 minutes
- TESS dataset (428 MB) — takes ~2–3 minutes

Expected output:
```
Downloading ravdess-emotional-speech-audio.zip to /content  100% 429M/429M
Downloading toronto-emotional-speech-set-tess.zip to /content  100% 428M/428M
All files downloaded as zip archives.
```

> 💡 If you see a Kaggle authentication error, your `kaggle.json` may be outdated.  
> Go back to Kaggle → Settings → API → Create New Token, and repeat Step 3.

---

### Step 5 — Run Cell 3: Extract Files

Click ▶ on **Cell 3**.

This unzips both datasets. Takes ~1–2 minutes.

Expected output:
```
RAVDESS Files: 2880
TESS Files: 5600
```

---

### Step 6 — Run Cell 4: Load Dataset

Click ▶ on **Cell 4**.

This scans all audio files and builds a dataframe with file paths and emotion labels.

Expected output:
```
Total audio files ready for processing: 8480
```

---

### Step 7 — Run Cell 5: Test Feature Extraction

Click ▶ on **Cell 5**.

This tests that feature extraction works on one file.

Expected output:
```
Testing feature extractor...
Feature shape: (180,)
```

If you see `Feature shape: (180,)` — everything is working correctly.

---

### Step 8 — Run Cell 7: Train the Model ⏳

Click ▶ on **Cell 7** (the big training cell).

> ⏳ **This takes approximately 20–25 minutes total:**
> - Feature extraction loop: ~8–10 minutes
> - Model training (30 epochs): ~12–15 minutes

You will see output like:
```
Processing features (this will take a few minutes)...
Training data shape: (6784, 180, 1)

Starting Training...
Epoch 1/30
106/106 ━━━━━━━━━━━━━━━━━━━━  12s - accuracy: 0.5598 - val_accuracy: 0.5896
Epoch 2/30
106/106 ━━━━━━━━━━━━━━━━━━━━  3s - accuracy: 0.7432 - val_accuracy: 0.7600
...
Epoch 30/30
106/106 ━━━━━━━━━━━━━━━━━━━━  3s - accuracy: 0.9817 - val_accuracy: 0.9575

Model training complete and saved!
```

**Final result: ~95.75% validation accuracy**

---

### Step 9 — Run Cells 8 and 9: Load Demucs Functions

Click ▶ on **Cell 8** (Demucs function definition — no visible output expected)  
Click ▶ on **Cell 9** (Sliding window function definition — no visible output expected)

These cells define the helper functions used in the demo.

---

### Step 10 — Run the Demo Cell: Test Your Own Audio 🎤

Click ▶ on **Cell 10** (the full pipeline demo).

A file upload dialog will appear:
1. Upload any `.wav` or `.mp3` audio file
2. The pipeline will:
   - Run Demucs to remove background noise (~30–60 seconds per minute of audio)
   - Segment long audio into 4-second overlapping chunks
   - Process all chunks in parallel
   - Display a per-segment emotion timeline
   - Show the dominant emotion with confidence score

Example output:
```
=== BVRIT Speech Emotion Recognition Demo ===
Upload an audio file (.wav format works best):

Isolating vocals with Demucs for: test_audio.wav...
Vocal isolation successful.
Segmenting audio and extracting features...
Processing 7 overlapping segments in parallel...

[ RESULT ] Dominant Emotion in Recording: ANGRY
```

---

### Step 11 — (Optional) Validate Against RAVDESS

Click ▶ on **Cell 11** (RAVDESS Validation Tester).

Upload any original RAVDESS `.wav` file (format: `03-01-05-01-01-01-01.wav`).  
The cell will decode the emotion from the filename and compare it to the model's prediction.

> Note: This only works with files using the original RAVDESS naming convention.  
> The file `OAF_burn_angry.wav` is a TESS file — it won't work here.

---

## Saving Your Model to Google Drive (Recommended)

To prevent losing your trained model when Colab disconnects, save it to Google Drive:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy model to Drive
import shutil
shutil.copy('/content/my_bvr_cnn_bilstm.keras',
            '/content/drive/MyDrive/my_bvr_cnn_bilstm.keras')
print("Model saved to Google Drive!")
```

### Loading the saved model in a future session

```python
from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras.models import load_model
model = load_model('/content/drive/MyDrive/my_bvr_cnn_bilstm.keras')
print("Model loaded from Drive!")
```

---

## Troubleshooting

### "Kaggle API error: 401 Unauthorized"
- Your `kaggle.json` is expired or wrong
- Go to Kaggle → Settings → API → Delete token → Create New Token
- Re-upload the new `kaggle.json` in Cell 1

### "CUDA out of memory"
- Reduce `batch_size` from 64 to 32 in Cell 7
- Or restart runtime and run again: Runtime → Restart and run all

### "Demucs output not found"
- The Demucs separation sometimes fails on very short files (<2 seconds)
- The code automatically falls back to using the original audio — this is expected

### "Feature shape is not (180,)"
- Check that all three feature extractions (MFCC, Chroma, Mel) ran without error
- Ensure `librosa` is installed: `!pip install librosa`

### "Session disconnected during training"
- Google Colab free tier disconnects after ~90 minutes of idle time
- If disconnected, re-run from Cell 4 onwards (datasets stay extracted)
- Save your model immediately after training to Google Drive (see above)

---

## Running Order Summary

| Cell | Action | Time |
|---|---|---|
| Cell 1 | Setup + upload kaggle.json | ~2 min |
| Cell 2 | Download datasets | ~5 min |
| Cell 3 | Extract files | ~2 min |
| Cell 4 | Load dataset | ~30 sec |
| Cell 5 | Test features | ~5 sec |
| Cell 7 | **Train model** ⏳ | ~20–25 min |
| Cell 8 | Load Demucs functions | ~1 sec |
| Cell 9 | Load segmentation functions | ~1 sec |
| Cell 10 | **Demo — test your audio** | ~1–5 min per file |
| Cell 11 | Optional RAVDESS validator | ~30 sec |

**Total: ~30–40 minutes from start to working demo**
