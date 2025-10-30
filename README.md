# Shodh_ai_assessment
1.Overview:
 This project explores and compares Supervised Learning and Offline Reinforcement Learning (RL) paradigms for predictive modeling using real-world data.
 The main objective is to understand how each learning method behaves, why certain results occur, and what trade-offs exist between classical supervised approaches  and RL-based decision optimization.

2. Environment Setup:
 Clone the repository :
  git clone https://github.com/kamalyld/Shodh_ai_assessment.git
  cd Shodh_ai_assessment
 Install dependencies:
  All libraries used in the notebook are listed in requirements.txt.
 Install them using:
  pip install -r requirements.txt
 If running in Google Colab, most dependencies (like TensorFlow and Scikit-learn) are pre-installed. Only d3rlpy needs explicit installation.
 Or you can just open this directly on colab :
  You can open and run this notebook directly in Google Colab:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamalyld/Shodh_ai_assessment/blob/main/Shodh_ai.ipynb)

3.Project Structure :
   Shodh_ai_assessment
    Shodh_ai.ipynb          -> Main Jupyter notebook (Colab)
    requirements.txt        -> Python dependencies
    README.md               -> Project documentation

4.Key files and where to look:
 Shodh_ai.ipynb — main notebook. Follow thiese sections and run them in the below order:
  Setup & dependencies
  Data loading & cleaning
  Exploratory Data Analysis (EDA)
  Feature engineering
  DL model definition, training, and evaluation
  Offline RL setup, training, and evaluation

5.Notebook structure & how to run individual sections:
 To reproduce only the DL experiment:
  Run cells through the end of "Feature engineering".
  Run the DL model training section.
 To reproduce only RL/OPE:
  Run dataset creation and anything required for offline RL dataset (MDPDataset).
  Run the d3rlpy section.
 
6.Methodology :
 Data Preprocessing :
  Missing values handled using SimpleImputer.
  Features scaled using StandardScaler to improve model convergence.
  Train-test split ensures balanced evaluation.
 Supervised Learning Model :
  Implemented using TensorFlow/Keras with a simple fully connected neural network.
  Optimized with EarlyStopping to avoid overfitting.
 Metrics evaluated:
  Accuracy
  ROC-AUC Score
  F1 Score
  Confusion Matrix
 Offline Reinforcement Learning Model :
  Implemented using d3rlpy with the Conservative Q-Learning (CQL) algorithm.
  Dataset transformed into an MDP format (MDPDataset) for offline RL training.
  Policy evaluation was performed using Fitted Q Evaluation (FQE) to estimate the Estimated Policy Value (EPV).
 Key Takeaway:
  Supervised models are better for prediction tasks, while RL models shine in decision-making tasks.
  The contrast in learning paradigms reveals how different algorithms perceive the same problem structure.

7.Evaluation Metrics:
 Supervised Model:
  Evaluated using F1-score and ROC-AUC to balance between precision and recall.
 RL Model:
  Evaluated using the Estimated Policy Value (EPV), which measures the expected return achieved under the learned policy.
 Visualization using Matplotlib and Seaborn helps interpret:
  Confusion matrices
  Reward convergence

8.Results Summary:
 The supervised model demonstrated strong generalization on static test data.
 The RL model, though more complex, offered a deeper understanding of long-term outcome optimization.
 Both approaches complement each other,supervised learning provides accurate state estimation, while RL provides adaptive decision logic.

9.Key Insights:
 Offline RL can effectively operate without online exploration, leveraging logged datasets.
 Deep Neural Networks (DNNs) serve as powerful function approximators for both paradigms.
 The why behind RL performance differences lies in its reward-driven optimization, not direct error minimization.

10.Future Improvements:
 Integrate hyperparameter tuning with Optuna or Ray Tune.
 Compare CQL with other RL algorithms (e.g., BCQ, TD3+BC).
 Deploy trained models via Streamlit or Flask for interactive visualization.
