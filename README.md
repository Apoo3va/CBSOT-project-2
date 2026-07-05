# Six Human Emotions Detection

A Natural Language Processing project that classifies text into one of **six human emotions** — Joy, Sadness, Anger, Fear, Love, and Surprise — using both classical Machine Learning and Deep Learning (LSTM) approaches, deployed as an interactive **Streamlit** web app.

> Type a sentence like *"I can't believe I got the job, this is amazing!"* and the app predicts the underlying emotion in real time.

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [Credits](#credits)
- [License](#license)

---

## Overview

This project tackles multi-class **text emotion classification** — given a sentence, predict which of six emotions it expresses. It was built as an end-to-end NLP pipeline:

1. Clean and preprocess raw text
2. Engineer features with TF-IDF
3. Train and compare multiple ML classifiers
4. Build a Deep Learning model (Embedding + LSTM) for improved performance
5. Serialize the best-performing pipeline
6. Serve predictions through a simple **Streamlit** UI

## Demo

The Streamlit app (`app.py`) provides a minimal UI:

1. Enter a sentence in the text box
2. Click **Predict**
3. View the predicted emotion label and the model's confidence score

## Dataset

- **File:** `train.txt`
- **Size:** 16,000 labeled sentences
- **Format:** `;`-separated text file — each line is `<sentence>;<emotion>`

```
i didnt feel humiliated;sadness
i am feeling grouchy;anger
i feel romantic too;love
ive been taking or milligrams or times recommended amount and ive fallen asleep a lot faster but i also feel like so funny;surprise
```

- **Classes (6):** `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`
- Class distribution is imbalanced — `joy` and `sadness` dominate, while `love` and `surprise` are minority classes (visualized in the notebook via count plots, length histograms, and word clouds per emotion).

## Approach

### 1. Text Preprocessing
Applied consistently across both the ML and DL pipelines:
- Remove non-alphabetic characters (regex)
- Lowercase normalization
- Tokenization
- Stopword removal (NLTK English stopwords)
- Stemming (`PorterStemmer`)

### 2. Machine Learning Pipeline
- **Feature extraction:** `TfidfVectorizer`
- **Models compared:**
  - Multinomial Naive Bayes
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVC)
- **Selected model:** Logistic Regression (chosen for its strong balance of accuracy, speed, and ease of deployment) — serialized to `logistic_regresion.pkl`

### 3. Deep Learning Pipeline (LSTM)
- Text converted to integer sequences via `one_hot` encoding and padded to a fixed length (`max_len = 300`, `vocab_size = 11000`)
- **Architecture:**

  | Layer | Output | Notes |
  |---|---|---|
  | Embedding | `input_dim=11000, output_dim=150, input_length=300` | |
  | Dropout | 0.2 | |
  | LSTM | 128 units | |
  | Dropout | 0.2 | |
  | Dense | 64, `sigmoid` | |
  | Dropout | 0.2 | |
  | Dense | 6, `softmax` | Output layer (6 emotion classes) |

- **Compilation:** `optimizer='adam'`, `loss='categorical_crossentropy'`, `metrics=['accuracy']`
- **Training:** Up to 10 epochs with `EarlyStopping` (`patience=2`)

## Results

### Classical ML — TF-IDF Accuracy Comparison (test set)

| Model | Accuracy |
|---|---|
| Multinomial Naive Bayes | 65.5% |
| **Logistic Regression** | **82.9%** |
| Random Forest | 84.8% |
| Support Vector Machine | 81.9% |

> Random Forest scored marginally highest, but Logistic Regression was selected for the deployed app for its lower footprint and faster inference.

### Deep Learning — LSTM (training accuracy)

The LSTM model showed strong learning over 10 epochs, improving from ~48% to **~97.8% training accuracy**:

| Epoch | Loss | Accuracy |
|---|---|---|
| 1 | 1.363 | 48.1% |
| 3 | 0.259 | 91.3% |
| 5 | 0.144 | 95.2% |
| 7 | 0.097 | 96.5% |
| 10 | 0.062 | **97.8%** |

*Note: These figures reflect training accuracy from the notebook run. Evaluate on a held-out test/validation split before drawing conclusions about generalization, and watch for overfitting given the gap in complexity vs. dataset size.*

## Project Structure

```
.
├── Emotions_Classification_using_ML_and_DL.ipynb   # Full pipeline: EDA, ML models, LSTM model
├── app.py                                          # Streamlit web app for inference
├── train.txt                                       # Training dataset (16,000 labeled sentences)
├── logistic_regresion.pkl                          # Trained Logistic Regression model
├── tfidf_vectorizer.pkl                            # Fitted TF-IDF vectorizer
├── label_encoder.pkl                               # Label encoder (emotion <-> class index)
├── vocab_info.pkl                                  # Vocab size & max sequence length used by the LSTM model
└── README.md
```

> **Note:** The trained LSTM model file (e.g. `model1.h5`) is referenced in the notebook's save step but is not included in this repository. Re-run the notebook's Deep Learning section to regenerate it if you want to serve the LSTM model instead of Logistic Regression.

## Installation

**Requirements:** Python 3.9+

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. Install dependencies:
   ```bash
   pip install streamlit numpy nltk scikit-learn==1.3.2
   pip install tensorflow==2.15.0   # only needed to run/retrain the LSTM notebook section
   pip install pandas seaborn matplotlib wordcloud   # only needed for the notebook / EDA
   ```

3. Download the NLTK stopwords corpus (done automatically on first run of `app.py`, or manually):
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Usage

### Run the Streamlit app
```bash
streamlit run app.py
```
This launches a local web server (default: `http://localhost:8501`). Enter text and click **Predict** to see the predicted emotion and confidence score.

### Explore or retrain the models
Open the notebook to see the full pipeline — EDA, preprocessing, ML model comparison, and LSTM training:
```bash
jupyter notebook Emotions_Classification_using_ML_and_DL.ipynb
```

## How It Works

`app.py` loads the three serialized artifacts (`logistic_regresion.pkl`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`) and, for each input:

1. Cleans the text (regex filtering, lowercasing, stopword removal, stemming)
2. Vectorizes it using the fitted TF-IDF vectorizer
3. Predicts a class with the Logistic Regression model
4. Decodes the class back into a human-readable emotion label
5. Displays the emotion and prediction confidence in the UI

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn (Logistic Regression, Naive Bayes, Random Forest, SVM, TF-IDF)
- **DL:** TensorFlow / Keras (Embedding + LSTM)
- **NLP:** NLTK (stopwords, stemming)
- **App/UI:** Streamlit
- **EDA/Visualization:** pandas, seaborn, matplotlib, wordcloud


##Author
**Apoorva Yadav**

## License

This project is licensed under the [MIT License](LICENSE).
