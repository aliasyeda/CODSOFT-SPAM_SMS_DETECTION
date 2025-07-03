# CODSOFT-SPAM_SMS_DETECTION

# 📩 Spam SMS Detection using Naive Bayes

This project focuses on building a machine learning model to detect **spam messages** using natural language processing (NLP). The goal is to classify SMS messages as **spam** or **ham** (not spam) using a dataset of labeled messages.



## 📂 Dataset

The dataset used is `spam.csv`, which contains SMS messages labeled as:

- **ham** – Legitimate (non-spam) messages
- **spam** – Unwanted or promotional messages



## 📌 Project Steps

1. **Load the dataset** using pandas
2. **Clean and preprocess** the data (drop unwanted columns, rename headers)
3. **Label encode** spam/ham messages (spam = 1, ham = 0)
4. **Split the data** into training and test sets
5. **Convert text to numbers** using TF-IDF vectorization
6. **Train the model** using Naive Bayes classifier
7. **Evaluate** the model with accuracy, precision, recall, F1-score, and confusion matrix



## 🤖 Algorithm Used

**Multinomial Naive Bayes**  
A simple yet powerful classification algorithm based on probability. It works well for text data by calculating the likelihood of each word in spam vs ham messages.



## 📈 Model Performance

- **Accuracy**: 97.39%
- **Confusion Matrix**:
[[965 0]
[ 29 121]]



- **Summary**:
- ✅ 965 legitimate (ham) messages correctly classified
- ✅ 121 spam messages correctly classified
- ❌ 29 spam messages missed (false negatives)
- ✅ 0 ham messages wrongly marked as spam (no false positives)



## 🧠 How Naive Bayes Works (Simple)

- Calculates **how often words appear** in spam and ham messages.
- Uses **probabilities** to decide whether a new message is spam or not.
- Assumes all words are **independent** (naive assumption).
- It’s **fast, simple, and works great for text classification** tasks.



## 🛠️ Tools & Libraries Used

- Python
- pandas
- scikit-learn
- TF-IDF Vectorizer
- Naive Bayes Classifier



## ✅ Results

The model shows **excellent accuracy and precision**, making it effective for real-world SMS spam detection systems.



## 📌 Future Improvements

- Use other models like SVM or Logistic Regression
- Add deep learning (LSTM, BERT)
- Build a web app with Streamlit or Flask






