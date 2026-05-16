# Relational Fraud Detection

A deep learning project focused on detecting financial fraud using Graph Neural Networks (GNNs). 

Traditional fraud detection models typically treat each transaction as an independent tabular record. However, fraud is inherently a relational phenomenon — fraudsters often share credit cards, device fingerprints, or email domains across multiple transactions. This project models the **IEEE-CIS Fraud Dataset** as a **Heterogeneous Graph** and applies a custom Attention-based GNN to learn network-level patterns, outperforming traditional tabular baselines.

## Model Architecture Overview

The core of this project is a **Heterogeneous Graph Attention Network**. 

Instead of just looking at the numerical features of a single transaction (like the dollar amount or time), the model relies on the *neighborhood* of the transaction in a graph.
- **Nodes**: There are multiple types of nodes. `transaction` nodes contain all the continuous features (V-features, amounts, etc.). `entity` nodes represent categorical identifiers like `card1`, `card4` (Visa/Mastercard), `P_emaildomain`, `DeviceType`, etc.
- **Edges**: Bipartite edges connect a `transaction` to the specific `entities` involved in that transaction. 
- **Message Passing**: During inference, information flows through the graph. If an `email_domain` node is heavily connected to known fraudulent transactions, that "risk" propagates through the edges to any new transaction connected to that same email node.

The project is structured into six distinct phases, built sequentially in Jupyter notebooks.

---

## 🏗️ The 6 Phases of the Pipeline

### Phase 01 — Data Preprocessing
The pipeline begins by loading the massive IEEE-CIS dataset. This phase focuses on cleaning the data, handling extensive missing values (NaNs), memory optimization (downcasting numerical types), and label-encoding categorical columns. The output is a clean, optimized tabular dataset ready for graph construction.

### Phase 02 — Exploratory Data Analysis (EDA)
An in-depth statistical analysis of the dataset. This phase includes:
- Analyzing the severe class imbalance (~3.5% fraud rate).
- Exploring distribution characteristics of continuous variables.
- Finding fraud rate patterns grouped by different categorical features (e.g., fraud rate by email domain or device type).
- Generating correlation heatmaps to understand feature redundancy.

### Phase 03 — Graph Construction
This phase transforms the flat tabular data into a rich mathematical graph structure.
- **Node extraction**: Extracting unique entities (cards, emails, devices) into their own node types.
- **Edge creation**: Creating `(transaction, has_card, card_node)` edge indices.
- **Feature matrices**: Assigning the continuous transaction features as the `x` tensor for the transaction nodes.
- The output is a `HeteroData` object compatible with PyTorch Geometric.

### Phase 04 — GNN Model
The core deep learning architecture, built from scratch in PyTorch.
- Implements a custom **Heterogeneous Feature Encoder** to map different node types into a common latent space.
- Implements **Heterogeneous Graph Attention Layers** that compute attention scores across different edge types, allowing the model to decide whether to weigh a shared "device" more heavily than a shared "email domain".
- The final layer acts on the updated `transaction` embeddings to predict a binary fraud probability.

### Phase 05 — Training
The training infrastructure for the GNN.
- Because the graph is massive (millions of edges), training utilizes `NeighborLoader` to sample localized subgraphs for mini-batch training.
- Addresses the 3.5% class imbalance using specialized loss functions (like Focal Loss or Class-Weighted Cross Entropy).
- Implements a rigorous training loop with learning rate scheduling, early stopping based on validation PR-AUC, and checkpointing the best model weights.

### Phase 06 — Evaluation & Results
The final phase evaluates the trained GNN against the holdout test set and compares it to traditional baselines (like MLPs or Gradient Boosting Classifiers).
- Computes hard metrics: **ROC-AUC** and **PR-AUC** (Average Precision).
- Generates comprehensive plots: ROC curves, Precision-Recall curves, and Confusion Matrices.
- Performs F1-score threshold tuning to find the optimal cutoff for fraud classification.
- Extracts the latent GNN embeddings and visualizes them using **t-SNE** to demonstrate how the model mathematically separates fraudulent networks from legitimate ones.