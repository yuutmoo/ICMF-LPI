# ICMF-LPI

## Overview
ICMF-LPI is a deep learning-based framework for predicting lncRNA-protein interactions (LPI) by leveraging heterogeneous information networks. The model incorporates a novel miRNA fusion mechanism, inter-view contrastive learning, and integrates miRNA as an intermediary between lncRNA and protein interactions.

## Dataset
The dataset for this project is organized into two primary directories, data1 and data2. The origin directory contains the information for each node, including sequences and species data for lncRNAs, miRNAs, and proteins. The order of the nodes corresponds to their biological information in the interaction matrices.
The `lncRNA_species.txt`, `miRNA_species.txt`, and `protein_species.txt` files in the `data1` directory contain the biological information for the nodes, including species data for lncRNAs, miRNAs, and proteins. In contrast, the `lncRNA sequence.txt`, `miRNA sequence.txt`, and `protein sequence.txt` files in the `data2` directory include the species names corresponding to each node. For detailed information, please refer to the `raid.v2_all_data.7z` file in the `data1` directory.

## Dependencies
This project requires the following libraries:
* PyTorch: 1.13.1
* DGL: 1.1.2+cu117

## Usage
To train the model, run the following command: Run
`python main.py`

## Code 

- **Layer.py**: Defines the layers used in the model, including graph-based layers or other neural network layers.
- **loss_function.py**: Implements the loss function(s) used for model training.
- **main.py**: The main entry point for running the model, where the dataset is loaded and the model is trained.
- **model.py**: Defines the model architecture, including layers, forward pass, and any specific network structure.
- **README.md**: The project documentation file containing detailed information about the project setup and usage.
- **utils.py**: A utility script with helper functions for data preprocessing, model support, etc.

## Data
- **lncRNA_GK_similarity_matrix.txt**: Similarity matrix for lncRNAs based on Gaussian Kernel (GK) features.
- **lncRNA_miRNA_interaction_matrix.csv**: Interaction matrix between lncRNAs and miRNAs.
- **lncRNA_protein_interaction_matrix.csv**: Interaction matrix between lncRNAs and proteins.
- **miRNA_GK_similarity_matrix.txt**: Similarity matrix for miRNAs based on Gaussian Kernel (GK) features.
- **protein_features.txt**: Contains feature vectors for proteins, used in training the model.
- **protein_GK_similarity_matrix.txt**: Similarity matrix for proteins based on Gaussian Kernel (GK) features.
- **protein_miRNA_interaction_matrix.csv**: Interaction matrix between proteins and miRNAs.
- **protein_species.txt**: Contains the biological information (species) for protein nodes.
- **proteinGO.txt**: Gene Ontology (GO) annotations for proteins.

