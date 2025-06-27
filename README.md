# 🧠 NLP Sentiment Analysis App

A simple and effective sentiment analysis web app built using **Natural Language Processing (NLP)** and **Machine Learning**. This project allows users to input text and receive predictions about the sentiment — **positive**, **negative**, or **neutral** — in real-time using a **Streamlit** interface.

---

## 📦 Features

* Uses ML/NLP techniques to classify sentiment
* Clean, interactive UI using Streamlit
* Trained on Twitter sentiment datasets
* Modular code structure for easy understanding and extensibility

---

## 🛠️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/akankshaydav/NLP-Sentiment-Analysis.git
   cd NLP-Sentiment-Analysis
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the dataset**

   Download or copy the `twitter_training.csv` and `twitter_validation.csv` into the `data/` directory.

---

## 🚀 Running the App

1. **Train the model**

   > ⚠️ Make sure the `.ipynb` is executed to save trained models in the `/models` directory.

   Open the notebook and run:

   ```bash
   jupyter notebook src/sentiment_analysis.ipynb
   ```

2. **Start the Streamlit App**

   ```bash
   streamlit run app.py
   ```

3. **Open your browser**

   By default, Streamlit runs at:
   [http://localhost:8501](http://localhost:8501)






## 📄 License

This project is licensed under the MIT License.


Would you like me to generate a Markdown file with this so you can directly add it to GitHub?
