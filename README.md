# CS4372 - Assignment 3

Google Colab: https://colab.research.google.com/drive/16JHyT1Ac75y45w7R-2Evp0LeRDOw-E-k?usp=sharing

Team: Elizaveta Makhonina & Avizeh Walji
Net ID: dal445515 & anw230000

This image classifier recognizes six natural scene classes including buildings, forest, glacier, mountain, sea, and street, and was built in Keras using a transfer learning approach. The architecture begins with a frozen ImageNet base for Stage 1 feature extraction, followed by Stage 2 fine tuning of the top layers using a reduced learning rate.

**Data:**
Source: Intel Image Classification on Kaggle: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
Loading: the notebook downloads the dataset with kagglehub.dataset_download("puneet6060/intel-image-classification"). 
Split: the provided seg_train folder is split 85% / 15% into train and validation (seed 42). The provided seg_test folder (3,000 images) is the test set, used once at the end. The unlabeled seg_pred folder isn't used.

**How to run:**
Open the Colab link above
Go to Runtime -> Change runtime type -> T4 GPU.
Go to Runtime -> Run all. A full run takes about 30 minutes on a free T4.
Outputs are saved in the Colab working directory (files panel):
tuning_log.csv: all experiments (tuning table)
best_model_curves.png: training/validation accuracy and loss for the best model, covering both stages
confusion_matrix.png: per-class test results
sample_predictions.png: 25 test images with true and predicted labels
models/exp*.keras: saved model for each experiment

**Requirements:**
Google Colab's default environment: Python 3, TensorFlow/Keras 3, kagglehub, numpy, pandas, matplotlib, scikit-learn.

Experiments:
Experiment #   Configuration
1	             MobileNetV2, frozen base, LR 1e-3, batch 32, 5 epochs, no augmentation
2              Exp 1 + data augmentation
3	             Exp 2 + fine-tune top 30 layers, LR 1e-5, 5 epochs
4	             Fine-tune top 60 layers
5	             Fine-tune LR 1e-4
6	             EfficientNetB0 base, fine-tune top 30
7	             ResNet50V2 base, fine-tune top 30
8	             Best base from 3/6/7, dropout 0.5, batch 64

The top model was selected based on validation performance across all 2 stage trials. To avoid data leakage and ensure fair testing, this final model was evaluated on the test set only once.
