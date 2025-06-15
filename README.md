#  Sentiment Analysis on Employee Reviews

This repository contains a machine learning pipeline to perform **sentiment analysis** on employee reviews. The goal is to classify text reviews as **positive (1)** or **negative (0)** based on their content and associated job roles.

---

##  Project Structure

```
sentiment-analysis/
├── data/                      # Training and testing datasets (.xlsx files)
├── src/                       # Core Python modules
│   ├── preprocess.py          # Text cleaning and feature engineering
│   ├── train.py               # Model training
│   ├── predict.py             # Model inference
├── main.py                    # Entry point to run training or prediction
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
```

---

##  Dataset Description

The dataset is divided into two Excel files:

- `train_ps3.xlsx`: Training dataset with 387 rows.
- `test_ps3.xlsx`: Test dataset with 130 rows.

Each row includes:
- `Job Role` – Categorical variable to be one-hot encoded.
- `Review Text` – Unstructured natural language input.
- `Sentiment` – Label (0 for Negative, 1 for Positive).

---

##  Features

- Text preprocessing (cleaning, stopwords removal, lemmatization).
- One-hot encoding of categorical job roles.
- TF-IDF vectorization for text features.
- Binary sentiment classification using a Neural Network (Keras).
- Easily extendable and modular codebase.
- CLI-style `main.py` to trigger training or inference.

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Install Dependencies

We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Add Data

Place your `train_ps3.xlsx` and `test_ps3.xlsx` inside the `data/` directory.

---

##  Usage

###  Train the Model

```bash
python main.py --mode train
```

- Trains a binary classifier on the training dataset.
- Saves the model as `sentiment_model.h5`.

###  Run Inference

```bash
python main.py --mode test
```

- Loads the trained model and predicts sentiment on the test set.
- Outputs predictions to the console (can be modified to save CSV or generate a report).

---

##  Model Architecture

Implemented using TensorFlow's Keras API:

```text
Input Layer: TF-IDF + Job Role Encoding
Dense Layer (128 units, ReLU)
Output Layer (1 unit, Sigmoid)
Loss: Binary Crossentropy
Optimizer: Adam
```

---

##  Example

Here’s what a cleaned and processed review might look like:

**Raw**:  
_"The management was not supportive and the environment was stressful."_

**Cleaned**:  
`management supportive environment stressful`

**Predicted Sentiment**:  
`0` (Negative)

---

##  Dependencies

All packages are listed in `requirements.txt`. Major ones include:

- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `tensorflow`
- `openpyxl` (for `.xlsx` support)

---

##  TODO

- Add precision/recall/F1-score reporting.
- Export predictions to CSV.
- Hyperparameter tuning.
- Model versioning and logging.
- Web or Streamlit UI.

---

##  Author

**Devendra Kumar Meena**  
[GitHub](https://github.com/ArcticPorch)

---

