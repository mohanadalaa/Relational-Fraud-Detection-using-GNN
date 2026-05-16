# 🕸️ Relational Fraud Detection using GNNs


A deep learning project focused on detecting financial fraud using Graph Neural Networks (GNNs). 

Traditional fraud detection models typically treat each transaction as an independent tabular record. However, fraud is inherently a relational phenomenon—fraudsters often share credit cards, device fingerprints, or email domains across multiple transactions. This project models the **IEEE-CIS Fraud Dataset** as a **Heterogeneous Graph** and applies a custom Attention-based GNN to learn network-level patterns, successfully outperforming traditional tabular baselines.

## 🚀 Key Results
* **Performance:** Achieved an **ROC-AUC of 0.8316**.
* **Improvement:** Outperformed a traditional tabular MLP baseline by **+6.6%**.
* **Architecture:** Engineered a custom 1.3M parameter Heterogeneous Graph Attention Network from scratch.

---

## 🧠 Model Architecture Overview

Instead of just looking at the numerical features of a single transaction (like the dollar amount or time), this model evaluates the *neighborhood* of the transaction within a graph structure.

* **Nodes**: Multiple node types exist. `transaction` nodes contain continuous features (amounts, time, etc.), while `entity` nodes represent categorical identifiers (e.g., `card1`, `P_emaildomain`, `DeviceType`).
* **Edges**: Bipartite edges connect a `transaction` to the specific `entities` involved.
* **Message Passing**: During inference, risk flows through the graph. If an `email_domain` node is heavily connected to known fraudulent transactions, that "risk" propagates through the edges to any new transaction connected to that same email node.

---

## 🏗️ The 6-Phase Pipeline

The project is structured into six distinct phases, built sequentially in Jupyter notebooks:

### 1️⃣ Data Preprocessing (`Phase01_Data_Preprocessing.ipynb`)
Loads the massive IEEE-CIS dataset. Focuses on cleaning data, handling extensive missing values (NaNs), memory optimization (downcasting numerical types), and label-encoding categorical columns. Outputs a clean tabular dataset ready for graph construction.

### 2️⃣ Exploratory Data Analysis (`Phase02_EDA.ipynb`)
In-depth statistical analysis, including:
* Analyzing severe class imbalance (~3.5% fraud rate).
* Exploring distributions of continuous variables.
* Identifying fraud rate patterns grouped by categorical features.
* Generating correlation heatmaps to understand feature redundancy.

### 3️⃣ Graph Construction (`Phase03_Graph_Construction.ipynb`)
Transforms flat tabular data into a rich mathematical graph:
* **Node extraction**: Isolating unique entities (cards, emails, devices).
* **Edge creation**: Building `(transaction, has_card, card_node)` edge indices.
* **Feature matrices**: Assigning continuous transaction features to transaction nodes.
* *Output:* A `HeteroData` object compatible with PyTorch Geometric.

### 4️⃣ GNN Model Construction (`Phase04_GNN_Model.ipynb`)
The core deep learning architecture, built from scratch:
* **Heterogeneous Feature Encoder**: Maps different node types into a common latent space.
* **Heterogeneous Graph Attention Layers**: Computes attention scores across different edge types (e.g., weighing a shared "device" differently than a shared "email").
* Predicts binary fraud probability based on updated transaction embeddings.

### 5️⃣ Model Training (`Phase05_Training.ipynb`)
Infrastructure for training on massive graphs:
* Utilizes **`NeighborLoader`** to sample localized subgraphs for mini-batch training.
* Addresses the 3.5% class imbalance using **Focal Loss** / Class-Weighted Cross Entropy.
* Implements a rigorous training loop with learning rate scheduling, early stopping (via validation PR-AUC), and checkpointing.

### 6️⃣ Evaluation & Results (`Phase06_Evaluation_Results.ipynb`)
Evaluates the GNN against a holdout test set and compares it to baselines (MLP, Gradient Boosting):
* Computes hard metrics: **ROC-AUC** and **PR-AUC**.
* Generates ROC curves, Precision-Recall curves, and Confusion Matrices.
* Performs F1-score threshold tuning.
* Extracts and visualizes latent embeddings using **t-SNE** to mathematically demonstrate separation between fraudulent and legitimate networks.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch, PyTorch Geometric (PyG)
* **Data Processing:** Pandas, NumPy, scikit-learn
* **Visualization:** Matplotlib, Seaborn


