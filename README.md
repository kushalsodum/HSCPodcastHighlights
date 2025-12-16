# Final Project - Hierarchical Sequence Classification for Predicting Podcast Highlights

This repository contains the code for training and evaluating Hierarchical Sequence Classification models for podcast highlight detection.

## Dataset
The dataset used is **Rhapsody**, which contains multimodal data for highlight detection. It can be accessed from its official repository: [https://github.com/younghanstark/rhapsody](https://github.com/Rhapsody-Dataset/Rhapsody).

## Source Code (`src/`)

The `src/` folder contains the following scripts:

### Data Processing
*   `preprocessData.py`: Helper functions for filtering highlights and calculating evaluation metrics (Precision, Recall, F1).

### Training Scripts
*   `train_rhapsody_linear.py`: Fine-tunes a Llama-3.2-1B model with a simple Linear classification head.
*   `train_rhapsody_lstm.py`: Fine-tunes the model using an **LSTM** layer on top of the LLM embeddings.
*   `train_rhapsody_bilstm.py`: Similar to the LSTM script but uses a **Bidirectional LSTM**.
*   `train_rhapsody_transformer.py`: Uses a **Transformer Encoder** layer instead of an LSTM for the classification head.

### Inference & Evaluation
*   `predict_rhapsody_linear.py`: Runs inference using the trained Linear model on the test set and saves the results.
*   `predict_rhapsody_lstm.py`: Runs inference using the trained LSTM model on the test set and saves the results.
*   `predict_rhapsody_bilstm.py`: Runs inference using the trained BiLSTM model on the test set and saves the results.
*   `predict_rhapsody_transformer.py`: Runs inference using the trained Transformer model on the test set and saves the results.
*   `find_best_threshold.py`: Analyzes the prediction results to find the optimal probability threshold for classification and calculates the Mean Average Precision (mAP).
*   `plot_results.py`: Generates bar charts comparing the performance (Precision, Recall, F1) of different models.

## Usage
1.  Clone Rhapsody dataset from HuggingFace
2.  Run a training script (e.g., `python src/train_rhapsody_lstm.py`).
3.  Run inference (e.g., `python src/predict_rhapsody_lstm.py`).
4.  Evaluate results (`python src/find_best_threshold.py`).
