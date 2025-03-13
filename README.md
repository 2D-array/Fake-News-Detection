

---

# 📰 Fake News Detection System

An interactive web app that detects whether a news article is **Fake** or **Real** using an ensemble of machine learning models. Built with **Streamlit**, it features real-time predictions, confidence scores, and visualized model votes.

![image](https://github.com/user-attachments/assets/497ff4db-7f78-44f4-a2a9-8ca8acbb1308)

*Replace this with an actual screenshot of your app.*

---

## 🚀 Features

- **Ensemble Learning**: Uses Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- **NLP-Powered**:  
  - Text preprocessing (lowercasing, cleaning).
  - **TF-IDF Vectorization** to convert text into meaningful features.
- **User-Friendly Dashboard**:
  - Confidence score for predictions.
  - Visual breakdown of each model's vote via **Plotly**.
- **Sample Inputs**: Predefined fake and real news samples.

---

## 📦 Installation

```bash
git clone https://github.com/2D-array/Fake-News-Detection.git
cd Fake-News-Detection
pip install -r requirements.txt
```

---

## 🖥️ Usage

```bash
streamlit run app.py
```

Then open the app in your browser and:
- Enter your own news text.
- Or test with sample fake/real news.
- See predicted label + model vote visualization.

---

## 🧠 Model Overview

### Dataset
Trained on the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-dataset) with:
- ~21k real news articles (Reuters).
- ~23k fake news articles (Politifact, GossipCop).

### Accuracy
| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 89%      |
| Random Forest        | 93%      |
| Gradient Boosting    | 92%      |
| **Ensemble**         | **95%**  |

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit UI + prediction logic
├── models/                 # .pkl model files
├── requirements.txt        # Dependencies
└── README.md               # You're reading it!
```

---

## 🛠️ Train Your Own Model

1. Place dataset in a `data/` folder.
2. Use `train_model.ipynb` (notebook) to:
   - Preprocess text
   - Train and save models (`.pkl`)
3. Load these models in `app.py`.

---
