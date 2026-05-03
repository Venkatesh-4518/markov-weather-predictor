# markov-weather-predictor
# 🌦️ Markov Chain Weather Predictor

## 📌 Overview
This project implements a Weather Prediction System using a Markov Chain model and core concepts from Linear Algebra. It predicts future weather conditions (Sunny, Rainy, Cloudy) based on probabilistic state transitions.

---

## 🚀 Features
- Predict weather for N days ahead  
- Simulate realistic day-by-day weather sequences  
- Compute steady-state distribution  
- Interactive Command Line Interface (CLI)  
- Visual probability representation using bars  

---

## 🧠 Mathematical Concepts Used
- Markov Chains (Memoryless property)  
- Transition Probability Matrix  
- Matrix Multiplication  
- Matrix Exponentiation (P^n)  
- Eigenvalues & Eigenvectors  
- Steady-State Distribution (πP = π)  

---

## ⚙️ How It Works
1. Weather states are represented as vectors (one-hot encoding).  
2. A transition matrix defines probabilities of moving between states.  
3. Future predictions are computed using:  

   X(t+n) = X(t) · P^n  

4. Long-term behavior is analyzed using eigenvector-based steady state.  

---

## 🛠️ Technologies Used
- Python  
- NumPy  

---

## 📂 Project Structure
```bash
Markov-Weather-Predictor/
│
├── main.py        # Main program (CLI-based)
├── README.md      # Project documentation
```

---

## ▶️ How to Run
1. Install dependencies:
```bash
pip install numpy
```

2. Run the program:
```bash
python main.py
```

---

## 📊 Example Output
- Probability of weather after N days  
- Most likely weather condition  
- Simulated weather sequence  
- Steady-state probabilities  

---

## 🎯 Applications
- Weather forecasting models  
- Stock market trend analysis  
- PageRank algorithm (Google Search)  
- Any probabilistic state-based system  

---

## 👨‍💻 Author
Venkatesh  
Computer Science Student  

---

## 📌 Conclusion
This project demonstrates how linear algebra and probability theory can be applied to build real-world prediction systems using Markov Chains.
