# 🧠 Disease Prediction Using Machine Learning

A machine learning-based system that predicts diseases based on user-input symptoms with high accuracy. Built using Python and deployed with Streamlit.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

---

## 📖 Overview

This project leverages machine learning algorithms to identify probable diseases based on symptoms entered by the user. The system is trained on a structured dataset and supports real-time prediction through a web-based interface.

---

## ✨ Features

- Symptom-based disease prediction  
- Interactive web interface using Streamlit  
- Trained on multiple ML models (Random Forest, Decision Tree, Naive Bayes)  
- Achieves up to 98% accuracy using Random Forest  
- Clean, user-friendly interface for real-time input

---

## 🧰 Tech Stack

- **Languages:** Python  
- **Libraries/Frameworks:**  
  - Scikit-learn  
  - Pandas  
  - NumPy  
  - Streamlit  
- **Tools:** Jupyter Notebook, VS Code

---

## ⚙️ How It Works

1. User selects symptoms from a list.
2. The system processes input and converts it into a format suitable for the model.
3. Trained machine learning models predict the most likely disease.
4. The result is displayed instantly through the web interface.

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/disease-detection-ml.git
cd disease-detection-ml

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
