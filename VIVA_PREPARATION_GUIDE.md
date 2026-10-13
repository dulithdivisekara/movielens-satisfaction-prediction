# Group 08 - IT2011 Final Evaluation & Viva Defense Preparation Guide
## Module: IT2011 - Artificial Intelligence and Machine Learning (2026)
### Assigned Dataset: MovieLens 32M (`ml-32m`)
### Project Title: Comparative Evaluation of 6 Machine Learning Classifiers for User Movie Satisfaction Prediction

---

## 👥 Group Structure & Correct Model Allocation

| Member Name | IT Number | Assigned Role & ML Model | Primary Metric & 5-Fold Variance |
| :--- | :--- | :--- | :--- |
| **Kaweesha P M G S** | Member 1 | **Member 1:** Logistic Regression (Probabilistic Linear Classifier) | F1: **0.7084**, ROC-AUC: **0.7850** ($s^2 = 2.15 \times 10^{-5}$) |
| **Divisekara A W D M D M B** | Member 2 | **Member 2:** Decision Tree Classifier (Axis-Aligned Rule-Based Model) | F1: **0.7182**, ROC-AUC: **0.7765** ($s^2 = 2.42 \times 10^{-5}$) |
| **Deshabhi R. G. S** | Member 3 | **Member 3:** Random Forest Classifier (Bagged Decision Tree Ensemble) | F1: **0.7385**, ROC-AUC: **0.8219** ($s^2 = 1.48 \times 10^{-5}$) |
| **Amadini G. G. A** | Member 4 | **Member 4:** Support Vector Machine (Linear vs Non-linear RBF Kernel) | F1: **0.7321**, ROC-AUC: **0.8040** ($s^2 = 1.76 \times 10^{-5}$) |
| **Hewage B H A S** | Member 5 | **Member 5:** K-Nearest Neighbors (KNN Instance-Based Lazy Learner) | F1: **0.7162**, ROC-AUC: **0.7914** ($s^2 = 2.20 \times 10^{-5}$) |
| **Perera D.A.A.N.S** | Member 6 | **Member 6:** Neural Network (Deep Multi-Layer Perceptron / MLP) | F1: **0.7394**, ROC-AUC: **0.8225** ($s^2 = 1.62 \times 10^{-5}$) |

---

## 🎯 Viva Opening Pitch (Any Member / Team Leader)
> *"Good morning/afternoon, Examiners. We are Group 08, presenting our project on the MovieLens 32M dataset.*
> *In commercial entertainment platforms (such as Netflix and YouTube), recommending content with a high probability of user satisfaction (rating $\ge 4.0$ stars) directly drives retention.*
> *Rather than testing arbitrary tasks, our group aligned all six members on the exact same benchmark classification challenge. This provides a direct, rigorous scientific comparison across six foundational AI/ML paradigms:*
> 1. *Linear Probabilistic Modeling: **Logistic Regression (Kaweesha)***
> 2. *Rule-Based Tree Partitioning: **Decision Tree (Divisekara)***
> 3. *Ensemble Bagging: **Random Forest (Deshabhi)***
> 4. *Geometric Maximum Margin: **Support Vector Machine (Amadini)***
> 5. *Instance-Based Neighborhoods: **K-Nearest Neighbors (Hewage)***
> 6. *Hierarchical Non-Linear Representation: **Deep MLP Neural Network (Perera)***
> *Every model was evaluated across parameter varieties using strict **5-Fold Stratified Cross-Validation**, with sample variance and standard deviation error-bar graphs. We are pleased to walk you through each notebook."*

---

## 📊 The "Which Variance is Used?" Examiner Question (CRITICAL)

Examiners love asking: *"You plotted error bars—what mathematical variance did you calculate, and what does it tell you?"*

### Exact Answer to Give:
1. **Mathematical Definition:**
   > *"We computed the **Unbiased Sample Variance ($s^2$)** across the $K=5$ validation folds with Bessel's correction ($N-1 = 4$ degrees of freedom):"*
   $$ s^2 = \frac{1}{K-1} \sum_{k=1}^K (x_k - \bar{x})^2 $$
   > *"where $x_k$ is the validation metric in fold $k$, and $\bar{x}$ is the cross-validation mean. The error bars on our graphs represent the **Sample Standard Deviation ($s = \sqrt{s^2}$)**."*

2. **Why It Matters for the Viva:**
   > *"The sample variance measures model stability and sensitivity to data partitioning. Because our sample variances across all 6 models remained strictly below $2.5 \times 10^{-5}$ (standard deviations $< 0.005$), we proved statistically that our models and feature scaling pipelines generalize consistently without high fold variance or lucky split bias."*

---

## 👤 Member-by-Member Detailed Viva Guide

---

### Member 1: Kaweesha P M G S
**Topic:** Supervised Classification via Logistic Regression
**Notebook:** `notebooks/Kaweesha_Logistic_Regression.ipynb`

#### 1. What to Show on Screen:
- Open `Kaweesha_Logistic_Regression.ipynb`.
- Show **Figure 1.1** (Correlation Heatmap), **Figure 1.2** (Top Standardized Log-Odds Coefficients), **Figure 1.3** (ROC Curves), and **Figure 1.4 & 1.5** (5-Fold CV Mean F1 with Variance Error Bars and Boxplot).

#### 2. What to Say (2-Minute Walkthrough):
> *"My model is **Logistic Regression**, which serves as our probabilistic linear benchmark. It maps linear combinations of standardized features into posterior probabilities via the sigmoid function:*
$$ P(y = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}} $$
> *I evaluated three regularization varieties by tuning the inverse penalty parameter $C$:*
> - *Variety A (Weak Regularization, $C=100.0$)*
> - *Variety B (Tuned Balanced L2 Regularization, $C=1.0$)*
> - *Variety C (Strong Regularization, $C=0.01$)*
> *Variety B achieved an optimal **F1-Score of 0.7084** and **ROC-AUC of 0.7850** with a fold variance of $s^2 = 2.15 \times 10^{-5}$.*
> *As shown in Figure 1.2, feature weights provide transparent interpretability: historical item quality (`movie_mean_rating`) and user generosity (`user_mean_rating`) contribute the highest positive log-odds weights to satisfaction."*

#### 3. Tough Questions & Answers:
- **Q: Why is feature scaling mandatory for Logistic Regression with L2 regularization?**
  - **A:** *"Because L2 regularization penalizes the sum of squared weights $\sum w_i^2$. If features have unequal scales, the weights of smaller-scale features must be artificially large, causing them to be penalized disproportionately by the regularizer."*
- **Q: What is the primary operational advantage of Logistic Regression?**
  - **A:** *"Sub-millisecond inference latency ($< 0.12$ ms per thousand items) and direct probability calibration, making it the ideal candidate for high-speed candidate filtering."*

---

### Member 2: Divisekara A W D M D M B
**Topic:** Supervised Classification via Decision Tree Classifier
**Notebook:** `notebooks/Divisekara_Decision_Tree.ipynb`

#### 1. What to Show on Screen:
- Open `Divisekara_Decision_Tree.ipynb`.
- Show **Figure 2.1 & 2.2** (Target balance & density), **Figure 2.3** (Top-10 Gini Importances), **Figure 2.4** (Overfitting Curve: Train vs Test Accuracy vs Depth), and **Figure 2.5 & 2.6** (5-Fold CV Mean F1 with Error Bars and Boxplot).

#### 2. What to Say (2-Minute Walkthrough):
> *"My model is a **Decision Tree Classifier**, which creates human-interpretable hierarchical decision rules by recursively minimizing Gini impurity:*
$$ G(m) = 1 - \sum_{k=0}^1 p_{m,k}^2 $$
> *I investigated the impact of tree depth across three varieties:*
> - *Variety A (Shallow Tree, $d=4$, min samples split 20)*
> - *Variety B (Tuned Pruned Tree, $d=8$, min samples split 10)*
> - *Variety C (Deep Tree, $d=16$, min samples split 2)*
> *In Figure 2.4, I demonstrated empirical overfitting: when depth increases to 16, training accuracy surges to 91% while testing accuracy collapses to 69%.*
> *Pruning at **depth 8 (Variety B)** delivered the optimal trade-off, achieving **F1-Score of 0.7182** and **ROC-AUC of 0.7765** with a fold variance of $s^2 = 2.42 \times 10^{-5}$."*

#### 3. Tough Questions & Answers:
- **Q: Why are Decision Trees prone to overfitting?**
  - **A:** *"Unconstrained trees have high variance and zero inductive bias toward smooth boundaries; they continue splitting until leaf nodes memorize individual noisy training points. Regularizing tree depth and minimum split sizes prevents this memorization."*
- **Q: What is an axis-aligned split limitation?**
  - **A:** *"Decision trees split one feature at a time parallel to feature axes. When the true decision boundary is diagonal, the tree must approximate it with a jagged staircase pattern, requiring deeper splits."*

---

### Member 3: Deshabhi R. G. S
**Topic:** Supervised Classification via Random Forest Classifier
**Notebook:** `notebooks/Deshabhi_Random_Forest.ipynb`

#### 1. What to Show on Screen:
- Open `Deshabhi_Random_Forest.ipynb`.
- Show **Figure 3.1 & 3.2** (Target distributions), **Figure 3.3** (Top-10 Gini Feature Importances), **Figure 3.4** (ROC Curves), and **Figure 3.5 & 3.6** (5-Fold CV Mean F1 with Variance Error Bars and Boxplot).

#### 2. What to Say (2-Minute Walkthrough):
> *"My model is a **Random Forest Classifier**, which directly resolves the high-variance limitation of Member 2's single Decision Tree through **Bootstrap Aggregation (Bagging)** and random feature subspace sampling.*
> *By ensembling $B$ de-correlated trees, the ensemble variance drops according to:*
$$ \text{Var} = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2 $$
> *I evaluated three varieties:*
> - *Variety A (Shallow: 50 trees, max depth 6)*
> - *Variety B (Tuned: 100 trees, max depth 12)*
> - *Variety C (Deep: 200 trees, max depth 20)*
> *Variety B emerged as our best tree-based model, achieving an **F1-Score of 0.7385** and **ROC-AUC of 0.8219**—a massive 2 percentage point gain over the single Decision Tree.*
> *Cross-validation sample variance dropped to $s^2 = 1.48 \times 10^{-5}$, proving the variance reduction power of bagging."*

#### 3. Tough Questions & Answers:
- **Q: How does feature subsampling de-correlate trees?**
  - **A:** *"If one feature is dominant (e.g. `movie_mean_rating`), every standard tree would split on it at the root, making all trees correlated. Randomly restricting split candidates to $m \approx \sqrt{p}$ features forces trees to explore alternative signals, lowering correlation $\rho$ and reducing total ensemble variance."*
- **Q: How do you interpret Random Forest?**
  - **A:** *"Through mean decrease in Gini impurity across all trees. `movie_mean_rating` and `user_mean_rating` contributed over 50% of the predictive split power."*

---

### Member 4: Amadini G. G. A
**Topic:** Supervised Classification via Support Vector Machine (Linear vs RBF SVM)
**Notebook:** `notebooks/Amadini_Support_Vector_Machine.ipynb`

#### 1. What to Show on Screen:
- Open `Amadini_Support_Vector_Machine.ipynb`.
- Show **Figure 4.1** (Correlation Matrix), **Figures 4.2 & 4.3** (Confusion Matrices: Linear vs RBF), and **Figures 4.4 & 4.5** (5-Fold CV Mean F1 with Error Bars and Boxplot).

#### 2. What to Say (2-Minute Walkthrough):
> *"My model is **Support Vector Machines (SVM)**, which constructs optimal hyper-planes that maximize the geometric margin separating classes:*
$$ \min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i $$
> *Because SVM computes Euclidean distances, feature standardization via `StandardScaler` is strictly enforced to prevent feature dominance.*
> *I compared three varieties:*
> - *Variety A: Linear SVM ($C=1.0$)*
> - *Variety B: Non-linear RBF Kernel SVM ($C=1.0, \gamma=\text{'scale'}$, infinite Hilbert space mapping)*
> - *Variety C: Non-linear RBF Kernel SVM ($C=5.0$, tighter margin penalty)*
> *The **Non-linear RBF Kernel (Variety B)** outperformed Linear SVM by over 2.5% in F1-score (**0.7321 vs 0.7064**), proving that movie satisfaction requires curved, non-linear decision boundaries.*
> *The 5-fold cross-validation variance across folds was $s^2 = 1.76 \times 10^{-5}$."*

#### 3. Tough Questions & Answers:
- **Q: What is the geometric role of the RBF Kernel?**
  - **A:** *"The Radial Basis Function kernel $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$ projects features into an infinite-dimensional space where non-linearly separable instances become linearly separable, enabling smooth circular or elliptic decision regions."*
- **Q: What is SVM's main operational bottleneck?**
  - **A:** *"Quadratic-to-cubic training complexity ($\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$). Training took 14.5 seconds on 12k instances, making it the slowest model to train in our group."*

---

### Member 5: Hewage B H A S
**Topic:** Supervised Classification via K-Nearest Neighbors (KNN)
**Notebook:** `notebooks/Hewage_K_Nearest_Neighbors.ipynb`

#### 1. What to Show on Screen:
- Open `Hewage_K_Nearest_Neighbors.ipynb`.
- Show **Figure 5.1 & 5.2** (Raw vs Scaled Variance), **Figure 5.3** (Test F1-Score vs Neighborhood Size $k$), **Figure 5.4** (ROC Curves), and **Figure 5.5 & 5.6** (5-Fold CV Mean F1 with Error Bars and Boxplot).

#### 2. What to Say (2-Minute Walkthrough):
> *"My model is **K-Nearest Neighbors (KNN)**, an instance-based non-parametric (lazy learning) algorithm. It makes predictions by locating the $k$ most geometrically similar interaction profiles in feature space and computing an inverse distance-weighted vote:*
$$ P(y = 1 \mid \mathbf{x}) = \frac{\sum_{i \in N_k(\mathbf{x})} w_i \cdot \mathbb{I}(y_i = 1)}{\sum_{i \in N_k(\mathbf{x})} w_i}, \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon} $$
> *I tuned neighborhood size and distance metrics across three varieties:*
> - *Variety A: $k=5$, uniform weights, Euclidean distance ($L_2$)*
> - *Variety B: $k=15$, distance-weighted, Euclidean distance ($L_2$)*
> - *Variety C: $k=35$, distance-weighted, Manhattan distance ($L_1$)*
> *In Figure 5.3, tuning $k$ revealed that $k=5$ suffers from high variance (noise sensitive), while $k=35$ over-smoothes boundaries. **$k=15$ with distance weighting (Variety B)** was optimal, achieving an **F1-Score of 0.7162** and **ROC-AUC of 0.7914** with fold variance $s^2 = 2.20 \times 10^{-5}$."*

#### 3. Tough Questions & Answers:
- **Q: Why is KNN called a 'lazy learner'?**
  - **A:** *"Because it performs zero inductive generalization during training (training time is virtually $0.1$ seconds). However, all computation is deferred to test time, where every query requires scanning the training set ($\mathcal{O}(N \cdot d)$), causing the highest inference latency in our group ($11.2$ ms)."*
- **Q: What is the benefit of distance-weighted voting over uniform voting?**
  - **A:** *"Uniform voting gives equal weight to all $k$ neighbors. Distance weighting gives higher influence to neighbors that are extremely close, reducing boundary distortion."*

---

### Member 6: Perera D.A.A.N.S
**Topic:** Supervised Classification via Neural Network (Multi-Layer Perceptron / MLP)
**Notebook:** `notebooks/Perera_Neural_Network_MLP.ipynb`

#### 1. What to Show on Screen:
- Open `Perera_Neural_Network_MLP.ipynb`.
- Show **Figure 6.1 & 6.2** (Standardized Activation Scaling), **Figure 6.3** (Binary Cross-Entropy Loss Convergence Curve), **Figure 6.4** (ROC Curves), and **Figure 6.5 & 6.6** (5-Fold CV Mean F1 with Variance Error Bars and Boxplot).

#### 2. What to Say (2-Minute Walkthrough):
> *"My model is a **Deep Learning Multi-Layer Perceptron (MLP)**. Neural networks learn continuous non-linear representations through layered affine transformations and ReLU activations:*
$$ \mathbf{h}^{(1)} = \text{ReLU}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}), \quad \hat{y} = \sigma(\mathbf{w}^{(L)T} \mathbf{h}^{(L-1)} + b^{(L)}) $$
> *I optimized cross-entropy loss with Adam and tested three architectural depths:*
> - *Variety A (Shallow: 1 layer of 64 neurons, $\alpha=0.01$)*
> - *Variety B (Tuned: 2 layers, $128 \times 64$ neurons, $\alpha=0.001$, Early Stopping)*
> - *Variety C (Deep: 3 layers, $128 \times 64 \times 32$ neurons, $\alpha=0.0001$)*
> *Variety B emerged as our overall group champion, achieving **F1-Score of 0.7394** and **ROC-AUC of 0.8225** with fold variance $s^2 = 1.62 \times 10^{-5}$.*
> *In Figure 6.3, the loss curve demonstrates smooth convergence, with early stopping preventing over-specialization."*

#### 3. Tough Questions & Answers:
- **Q: Why does MLP outperform Linear models and single Trees?**
  - **A:** *"Because dense layers learn distributed feature representations. Combinations of user generosity and genre features are transformed into latent hidden vectors, capturing complex interactions that linear models miss."*
- **Q: How does MLP compare to Random Forest?**
  - **A:** *"MLP achieved slightly higher ROC-AUC (0.8225) and three times faster inference per batch ($1.1$ ms vs $2.9$ ms). However, Random Forest provides superior Gini interpretability."*

---

## 🏆 Group Conclusion & Master Comparison (All 6 Members)
> *"In summary, our group conducted an apples-to-apples comparison of 6 machine learning paradigms on MovieLens 32M satisfaction prediction:*
> 1. *Ranking Champions: **Deep MLP (Perera: F1=0.7394)*** and ***Random Forest (Deshabhi: F1=0.7385)***.
> 2. *Speed Champions: **Logistic Regression (Kaweesha: 0.12 ms)*** and ***Decision Tree (Divisekara: 0.15 ms)***.
> 3. *Every model achieved minimal 5-fold cross-validation variance ($s^2 < 2.5 \times 10^{-5}$), proving statistical robustness.*
> 4. *Production Recommendation: A two-tier architecture using Logistic Regression for rapid 5 ms candidate filtering followed by Deep MLP or Random Forest for top-10 personalized ranking.*
> *Thank you, Examiners!"*
