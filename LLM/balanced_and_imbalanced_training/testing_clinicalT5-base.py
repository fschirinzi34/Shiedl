from pathlib import Path
import sys
import csv
import argparse
import torch
import pandas as pd
import os
from datetime import datetime

from torch import seed

from custom_libs_clinical_t5_base.data_handler import DataHandler
from custom_libs_clinical_t5_base.pii_data_loader import PIIDataLoader
from custom_libs_clinical_t5_base.ClinicalT5_ModelTrainer import ClinicalT5_ModelTrainer


### Constants
METHODOLOGY = 1
RANDOM_SEED=123 # Fixed seed for reproducible results
WEIGHT_MODEL_PATH= r'results_train/ClinicalT5-base_model_tte.pt' # Model from Hugging Face
MODEL_NAME='hossboll/clinical-t5' # Model from Hugging Face
SAVE_FOLDER='results_train' # Path where to save the data

### Functions
def save_results_to_csv(results):
    """Simple version that saves directly to current directory"""

    try:
        filename = f"testing_result_{SUFFIX}.csv"

        # Create results directory if it doesn't exist
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        filepath = os.path.join(SAVE_FOLDER, filename)

        print(f"Saving results to: {filepath}")
        df = pd.DataFrame(results)

        file_esiste = os.path.exists(filepath)
        df.to_csv(filepath, mode="a", index=False, header=not file_esiste)

        if os.path.exists(filepath):
            print(f"File created successfully: {filepath}")
            return filepath
        else:
            print("File was not created!")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def evaluation(device, num_labels=2, seed=10):

    # Set deterministic behavior for reproducible results
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(RANDOM_SEED)

    # Initialize data handler and load the test dataset
    data_handler = DataHandler(DATASET_PATH, train_size=0.8, valid_size=0, test_size=0.2, seed=seed)
    data_handler.load_data()
    data_handler.clean_data()

    # Split data into training, validation, and test sets
    if METHODOLOGY == 1:
        _, _, _, _, test_texts_gen, test_labels_gen = data_handler.split_data(
            dataset_type=True)
    else:
        _, _, _, _, test_texts_gen, test_labels_gen = data_handler.split_data(
            n_rows=1000, dataset_type=True)

    _, _, _, _, test_texts_temist, test_labels_temist = data_handler.split_data(
            n_rows=1000, dataset_type=False)

    results1 = doEvaluation(device, test_texts_gen, test_labels_gen, seed,  "GENERATED", num_labels)
    print(results1)
    results2 = doEvaluation(device, test_texts_temist, test_labels_temist, seed, "ZENODO", num_labels)
    print(results2)


def doEvaluation(device, test_texts, test_labels, passed_seed, datasetName, num_labels=2):
    # Create data loaders for batch processing
    data_loader = PIIDataLoader(MODEL_NAME, test_texts=test_texts, test_labels=test_labels, batch_size=100)
    test_loader = data_loader.get_specific_dataloader('test')

    # Initialize ClinicalT5 model trainer (corretto il nome della classe)
    model_trainer = ClinicalT5_ModelTrainer(
        model_name=MODEL_NAME,
        save_folder=SAVE_FOLDER,
        device=device,
        weight_model_path=WEIGHT_MODEL_PATH,
        num_labels=num_labels,  # Aggiunto parametro num_labels
    )

    # Evaluate the trained model on the test set
    test_results = model_trainer.evaluate(test_loader)

    # Make sure the folder exists
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Prepare results for CSV
    results = [{
        'Dataset': datasetName,
        'seed': passed_seed,
        'accuracy': test_results.get('accuracy', 0),
        'precision': test_results.get('precision', 0),
        'recall': test_results.get('recall', 0),
        'f1_score': test_results.get('f1_score', 0),
        'auc_roc': test_results.get('auc_roc', 0),
        'average_loss': test_results.get('loss', 0),
    }]

    # Save results to CSV
    save_results_to_csv(results)
    return results

def arg_commandline():
    parser = argparse.ArgumentParser(description="ClinicalT5 model testing")

    parser.add_argument('-b', '--standard', action='store_true', help='enable standard test split')
    parser.add_argument('-k', '--kfold', action='store_true', help='enable test from K-fold cross-validation training')
    parser.add_argument('-s', '--strkfold', action='store_true', help='enable test from stratified k-fold cross-validation training')

    parser.add_argument('--dataset', type=str, default='Dataset\\', help='Path to the dataset file')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of classification labels')  # Nuovo parametro
    parser.add_argument('--n_splits', type=int, default=2, help='Number of splits in kfold')  # Nuovo parametro

    return parser.parse_args()

### Main
def main():
    args = arg_commandline()

    global DATASET_PATH, SUFFIX
    DATASET_PATH = args.dataset

    print(f"\n\nDataset utilizzato: {Path(DATASET_PATH).name}\n\n")

    if args.strkfold:
        SUFFIX=f"stratified_kfold_{args.n_splits}splits"
    elif args.kfold:
        SUFFIX=f"kfold_{args.n_splits}splits"
    else:
        SUFFIX=f"standard"

    with open(f"results_train\\training_results_{SUFFIX}.csv", mode="r", encoding="utf-8") as file:

        reader = csv.DictReader(file)
        rows = list(reader)

        if not rows:
            print("Error: CSV file is empty!!")
            return None

        # Recuperiamo l'ultima riga
        last_row = rows[-1]

        # Estraiamo il valore della colonna 'seed' e salviamo nella variabile
        seed_value = int(last_row["seed"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using model: {WEIGHT_MODEL_PATH}")
    print(f"Number of labels: {args.num_labels}")

    print()
    print("Starting evaluation...")
    print(seed_value)
    evaluation(device=device, num_labels=args.num_labels, seed=seed_value)

if __name__ == "__main__":
    main()
