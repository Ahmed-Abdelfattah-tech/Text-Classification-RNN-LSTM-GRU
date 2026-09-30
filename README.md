# AG News Text Classification: Simple RNN vs. LSTM vs. GRU

## 📌 Project Overview

This project solves a **4-class text classification problem** on the **AG News** dataset using recurrent neural networks.

Three recurrent architectures are built, trained, and compared under the same data split and the same embedding setup:

1. **Simple RNN**
2. **LSTM** (Bidirectional)
3. **GRU** (Bidirectional)

The goal is to compare a basic recurrent layer against gated architectures (LSTM and GRU), which were designed to solve the vanishing-gradient limitation of Simple RNNs on longer sequences.

The best model, **LSTM, reaches 91.84% test accuracy**, closely followed by GRU at 91.80%, both clearly ahead of the Simple RNN baseline at 90.34%.

\---

## 🎯 Objective

Classify AG News articles into one of **4 topic categories** using only the article's title and description, and compare three recurrent architectures on:

* Test accuracy and test loss
* Training stability (loss curves, gap between training and validation)
* How much gated units (LSTM/GRU) improve on a plain RNN

\---

## 📊 Dataset

The **AG News** dataset is loaded directly from the `CharCnn\_Keras` GitHub repository as CSV files (no manual download needed — the notebook fetches it from the URLs itself).

|Set|Rows|Description|
|-|-:|-|
|Training|108,000|90% of the original training set|
|Validation|12,000|10% of the original training set (stratified)|
|Test|7,600|Official AG News test set, used only for final evaluation|

```python
train\_test\_split(train\_df, test\_size=0.10, random\_state=42, stratify=train\_df\["label"])
```

### Classes

The dataset has **4 balanced topic classes** (originally labeled 1–4, remapped to 0–3):

* World
* Sports
* Business
* Sci/Tech

### Text Field

Each example's `title` and `description` columns are concatenated into a single `text` column, which is the only input to the models.

\---

## ⚙️ Preprocessing

1. **Custom text standardization:** lowercasing, replacing `<br />` tags, and stripping punctuation.
2. **Text vectorization** with Keras' `TextVectorization` layer:

   * Vocabulary size: **20,000 tokens** (`MAX\_TOKENS`)
   * Sequence length: **200 tokens** (`SEQ\_LEN`), with padding/truncation
   * The vectorizer's vocabulary is learned **only on the training text**, then applied to validation and test sets — avoiding data leakage.
3. **`tf.data` pipeline:** the raw text/label pairs are wrapped in `tf.data.Dataset`, vectorized once, then batched (`BATCH\_SIZE = 64`) and cached for efficient training.
4. **Reproducibility:** `keras.utils.set\_random\_seed(1337)`, plus explicit NumPy and TensorFlow seeds.

\---

## 🤖 Models

All three models share the same input pipeline and a **trainable embedding layer** (not pretrained), so any performance difference comes from the recurrent layer itself.

### Shared Embedding

```python
layers.Embedding(input\_dim=MAX\_TOKENS, output\_dim=EMBED\_DIM, mask\_zero=True)
```

* `MAX\_TOKENS = 20000`, `EMBED\_DIM = 100` → 2,000,000 embedding parameters in every model.
* `mask\_zero=True` so the recurrent layers ignore padding tokens.

### 1️⃣ Simple RNN

```text
Input (200,) → Embedding (200, 100) → SimpleRNN(64) → Dropout(0.5) → Dense(4, softmax)
```

* **Total parameters:** 2,010,820 (7.67 MB), all trainable

### 2️⃣ LSTM (Bidirectional)

```text
Input (200,) → Embedding (200, 100) → Bidirectional(LSTM(64)) → Dropout(0.5) → Dense(4, softmax)
```

* **Total parameters:** 2,084,996 (7.95 MB), all trainable

### 3️⃣ GRU (Bidirectional)

```text
Input (200,) → Embedding (200, 100) → Bidirectional(GRU(64)) → Dropout(0.5) → Dense(4, softmax)
```

* **Total parameters:** 2,064,260 (7.87 MB), all trainable

### Training Setup (all three models)

|Setting|Value|
|-|-|
|Optimizer|Adam|
|Loss|Sparse Categorical Crossentropy|
|Batch size|64|
|Max epochs|10|
|Early Stopping|monitor `val\_accuracy`, patience 3, `restore\_best\_weights=True`|

|Model|Epochs run|Best epoch restored|
|-|-:|-:|
|Simple RNN|5|2|
|LSTM|4|1|
|GRU|4|1|

All three models stopped early: validation accuracy peaked in the **first or second epoch**, after which validation loss started rising while training accuracy kept climbing — a clear sign of overfitting on later epochs.

\---

## 📈 Model Performance

All models were evaluated on the same held-out test set (7,600 samples).

|Model|Train Acc|Val Acc|Val Loss|Test Acc|Test Loss|
|-|-:|-:|-:|-:|-:|
|Simple RNN|0.9544|0.9009|0.4033|0.9034|0.3306|
|**LSTM**|**0.9599**|**0.9097**|**0.3098**|**0.9184**|**0.2506**|
|GRU|0.9624|0.9047|0.3171|0.9180|0.2532|

### Training Curves

**Simple RNN**

!\[RNN accuracy](accuracy\_curve\_RNN.png)

!\[RNN loss](loss\_curve\_RNN.png)

**LSTM (Bidirectional)**

!\[LSTM accuracy](accuracy\_curve\_LSTM.png)

!\[LSTM loss](loss\_curve\_LSTM.png)

**GRU (Bidirectional)**

!\[GRU accuracy](accuracy\_curve\_GRU.png)

!\[GRU loss](loss\_curve\_GRU.png)

\---

## 🔍 Analysis

* **Gated units beat the Simple RNN**, but by a moderate margin here (about 1.5–1.6 points of test accuracy), not the dramatic gap sometimes seen on much longer sequences. With `SEQ\_LEN = 200` and short news snippets, the vanishing-gradient problem that LSTM/GRU are designed to solve is less severe than it would be on longer documents.
* **LSTM and GRU perform almost identically** (91.84% vs. 91.80% test accuracy, both within 0.003 loss of each other), while GRU uses **about 20,000 fewer parameters** than LSTM. This matches the general pattern that GRU often reaches comparable accuracy to LSTM more cheaply.
* **All three models overfit quickly.** Every model's best validation epoch came within the first two epochs, and training accuracy kept rising afterward while validation loss increased — visible clearly in the loss curves. This suggests the models have enough capacity to memorize the training set well before generalization peaks.
* **Bidirectionality helps LSTM/GRU see future context** (words later in the sentence), which the unidirectional Simple RNN does not have — this is a likely contributor to their better performance, separate from the gating mechanism itself.

\---

## 🔭 Limitations \& Future Improvements

* **No pretrained embeddings:** the embedding layer is trained from scratch. Using pretrained vectors (GloVe, Word2Vec) or a pretrained transformer (e.g. DistilBERT) would likely improve accuracy and reduce overfitting.
* **Early stopping on `val\_accuracy` only:** all models stopped within the first few epochs; tracking `val\_loss` alongside, or using a smaller learning rate with more patience, might let the models train longer before overfitting takes over.
* **No hyperparameter tuning:** unit sizes (64), dropout (0.5), and sequence length (200) were fixed across all three models for a fair comparison, but were not individually tuned.
* **No regularization beyond Dropout:** techniques like recurrent dropout or L2 weight regularization were not explored.
* **Simple RNN is unidirectional while LSTM/GRU are bidirectional:** this makes the comparison slightly uneven, since bidirectionality itself is a contributing factor, not just the gating mechanism. A bidirectional Simple RNN would isolate the gating effect more cleanly.
* **Confusion matrix / per-class performance** was not analyzed — some categories (e.g. Business vs. Sci/Tech) are more prone to being confused than others.

\---

## 🧠 Key Deep Learning / NLP Concepts Demonstrated

* Text classification with Recurrent Neural Networks
* Text standardization and cleaning
* Text vectorization with Keras `TextVectorization`
* Trainable word embeddings with padding masking (`mask\_zero`)
* `tf.data` pipelines (map, cache, batch, prefetch)
* Simple RNN, LSTM, and GRU architectures
* Bidirectional recurrent layers
* Vanishing-gradient intuition and how gating mechanisms address it
* `EarlyStopping` with `restore\_best\_weights`
* Overfitting analysis with accuracy and loss curves
* Model comparison on a held-out test set

\---

## 🛠️ Technologies

* **Python**
* **TensorFlow / Keras**
* **NumPy**, **Pandas**
* **Matplotlib**, **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

\---

## 📁 Project Structure

```text
AGNews-Text-Classification-RNN-LSTM-GRU/
│
├── Text\_Classification\_Using\_RNN\_LSTM\_GRU\_Models.ipynb
├── images/
│   ├── accuracy\_curve\_RNN.png
│   ├── loss\_curve\_RNN.png
│   ├── accuracy\_curve\_LSTM.png
│   ├── loss\_curve\_LSTM.png
│   ├── accuracy\_curve\_GRU.png
│   └── loss\_curve\_GRU.png
├── README.md
├── requirements.txt
└── .gitignore
```

\---

## 🚀 How to Run

### 1\. Clone the repository

```bash
git clone https://github.com/Ahmed-Abdelfattah-tech/AGNews-Text-Classification-RNN-LSTM-GRU.git
cd AGNews-Text-Classification-RNN-LSTM-GRU
```

### 2\. Install dependencies

```bash
pip install -r requirements.txt
```

### 3\. Open the notebook

```text
Text\_Classification\_Using\_RNN\_LSTM\_GRU\_Models.ipynb
```

The AG News dataset is downloaded automatically from GitHub on the first run — no manual download needed.

Run the cells sequentially to reproduce the preprocessing, training, and comparison of all three models.

\---

## 📌 Key Takeaways

This project compares three recurrent architectures on the same text classification task:

**Simple RNN (90.34%) → GRU (91.80%) → LSTM (91.84%)** test accuracy.

Gated architectures (LSTM, GRU) outperformed the plain RNN, and did so almost identically to each other despite GRU having fewer parameters. All three models showed early overfitting, with validation performance peaking in the first or second epoch — a useful reminder that recurrent models on short-text tasks can converge (and overfit) very quickly.

> \*\*Best result: 91.84% test accuracy on AG News with a Bidirectional LSTM\*\*

\---

## 👤 Author

**Ahmed Abdelfattah**



