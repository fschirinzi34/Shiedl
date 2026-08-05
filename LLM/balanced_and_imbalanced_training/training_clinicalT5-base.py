from datetime import datetime

import torch
import argparse
import pandas as pd
import os
import time
from custom_libs_clinical_t5_base.data_handler import DataHandler
from custom_libs_clinical_t5_base.pii_data_loader import PIIDataLoader
from custom_libs_clinical_t5_base.ClinicalT5_ModelTrainer import ClinicalT5_ModelTrainer


### Constants
METHODOLOGY = 1 # 1 for imbalanced training, 0 for balanced training
RANDOM_SEED=123 # Fixed seed for reproducible results
MODEL_NAME='hossboll/clinical-t5' # Model from Hugging Face
MODEL_RES='ClinicalT5-base_model'
OPTIMIZER_RES='optimizer'
SAVE_FOLDER='results_train' # Path where to save the data

### Functions
def save_results_to_csv(results, method_name):
    """Add results to a already created CSV file."""

    try:
        filename = f"training_results_{method_name}.csv"

        # Make new dir if dir does not exist
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        filepath = os.path.join(SAVE_FOLDER, filename)

        df = pd.DataFrame(results)

        # Add rows if file exist, else it will be created first
        if os.path.exists(filepath):
            df.to_csv(filepath, mode="a", header=False, index=False)
        else:
            df.to_csv(filepath, mode="w", header=True, index=False)

        print(f"Results saved to: {filepath}")
        return filepath

    except Exception as e:
        print(f"Error while saving results: {e}")
        return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def training_test_validation(num_epochs, device, num_labels=2):
    """ Performs standard train-test-validation split training on the dataset """


    # Set deterministic behavior for reproducible results
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(RANDOM_SEED)

    seed = int(time.time())

    # Initialize data handler and load the test dataset
    data_handler = DataHandler(DATASET_PATH, train_size=0.75, valid_size=0.0, test_size=0.25, seed=seed) # path of our datasets
    data_handler.load_data()
    data_handler.clean_data()

    # Split data into training, validation, and test sets
    if METHODOLOGY == 1:
        train_texts_gen, train_labels_gen, valid_texts_gen, valid_labels_gen, test_texts_gen, test_labels_gen = data_handler.split_data(dataset_type=True)
    else:
        train_texts_gen, train_labels_gen, valid_texts_gen, valid_labels_gen, test_texts_gen, test_labels_gen = data_handler.split_data(n_rows=1000, dataset_type=True)

    train_texts_temist, train_labels_temist, valid_texts_temist, valid_labels_temist, test_texts_temist, test_labels_temist = data_handler.split_data(n_rows=1000, dataset_type=False)

    train_texts = train_texts_gen + train_texts_temist
    train_labels = train_labels_gen + train_labels_temist

    valid_texts = (
            (valid_texts_gen if valid_texts_gen is not None else []) +
            (valid_texts_temist if valid_texts_temist is not None else [])
    )

    valid_labels = (
            (valid_labels_gen if valid_labels_gen is not None else []) +
            (valid_labels_temist if valid_labels_temist is not None else [])
    )

    test_texts = test_texts_gen + test_texts_temist
    test_labels = test_labels_gen + test_labels_temist

    # Create data loaders for batch processing
    data_loader = PIIDataLoader(train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels)
    train_loader, valid_loader, test_loader = data_loader.get_dataloader()

    # Initialize ClinicalT5 model trainer
    model_trainer = ClinicalT5_ModelTrainer(
        model_name=MODEL_NAME,
        save_folder=SAVE_FOLDER,
        device=device,
        num_labels=num_labels,
        num_epochs=num_epochs
    )

    # Train the model using training and validation sets
    model_trainer.train(train_loader, None)

    # Evaluate the trained model on the test set
    test_results = model_trainer.evaluate(test_loader)

    # Make sure the folder exists
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Save the model and optimizer state with the folder prefix
    model_trainer.save_model(
        os.path.join(SAVE_FOLDER, MODEL_RES + f'_tte.pt'),
        os.path.join(SAVE_FOLDER, OPTIMIZER_RES + f'_tte.pt')
    )

    # Generate and display training metrics plots
    # model_trainer.plot_metrics()

    # Prepare results for CSV


    results = [{
        'method': 'standard_split',
        'methodology': '750 Temist + 75% Generated' if METHODOLOGY==1 else "750 Temist + 750 Generated",
        'seed': seed,
        'fold': 1,
        'epochs': num_epochs,
        'test_accuracy': test_results.get('accuracy', 0),
        'precision': test_results.get('precision', 0),
        'recall': test_results.get('recall', 0),
        'f1_score': test_results.get('f1_score', 0),
        'auc_roc': test_results.get('auc_roc', 0),
        'average_loss': test_results.get('loss', 0),
        'training_time_min': test_results.get('training_time_min', 0)
    }]

    # Save results to CSV
    save_results_to_csv(results, 'standard')
    return results

def kfold_training(n_splits, num_epochs, device, num_labels=2):
    """ Performs K-Fold cross-validation training on the dataset """

    # Set deterministic behavior for reproducible results
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(RANDOM_SEED)

    # Initialize data handler with K-fold configuration
    data_handler = DataHandler(DATASET_PATH, n_splits=n_splits)
    data_handler.load_data()
    data_handler.clean_data()

    # Store evaluation results from each fold
    fold_results = []

    # Iterate through each fold of the cross-validation
    for fold, (train_texts, train_labels, test_texts, test_labels) in enumerate(data_handler.kfold_split_data()):
        print(f"Training on fold {fold + 1}/{n_splits}...")

        # Create data loaders for current fold (no validation set in K-fold)
        data_loader = PIIDataLoader(train_texts, train_labels, [], [], test_texts, test_labels)
        train_loader, _, test_loader = data_loader.get_dataloader()

        # Initialize a fresh model trainer for each fold (corretto il nome della classe)
        model_trainer = ClinicalT5_ModelTrainer(
            model_name=MODEL_NAME, 
            save_folder=SAVE_FOLDER, 
            device=device, 
            num_labels=num_labels,  # Aggiunto parametro num_labels
            num_epochs=num_epochs
        )

        # Train the model on current fold's training data
        model_trainer.train(train_loader, None) # No validation loader for K-fold

        # Evaluate the model on current fold's test data
        test_results = model_trainer.evaluate(test_loader)
        
        # Prepare fold results for CSV
        fold_result = {
            'method': 'kfold',
            'fold': fold + 1,
            'epochs': num_epochs,
            'test_accuracy': test_results.get('accuracy', 0),
            'precision': test_results.get('precision', 0),
            'recall': test_results.get('recall', 0),
            'f1_score': test_results.get('f1_score', 0),
            'auc_roc': test_results.get('auc_roc', 0),
            'average_loss': test_results.get('loss', 0),
            'training_time_min': test_results.get('training_time_min', 0)
        }
        fold_results.append(fold_result)

        # Make sure the folder exists
        os.makedirs(SAVE_FOLDER, exist_ok=True)

        # Save the model and optimizer state with the folder prefix
        model_trainer.save_model(
            os.path.join(SAVE_FOLDER, MODEL_RES + f'_fold_{fold + 1}.pt'),
            os.path.join(SAVE_FOLDER, OPTIMIZER_RES + f'_fold_{fold + 1}.pt')
        )

    print(f"Cross-validation completed. Results collected for {len(fold_results)} folds.")

    # Calculate average results across all folds
    avg_results = calculate_average_results(fold_results)
    fold_results.append(avg_results)

    # Generate and display training metrics plots
    model_trainer.plot_metrics()

    # Save results to CSV
    save_results_to_csv(fold_results, f'kfold_{n_splits}splits')
    return fold_results

def stratified_kfold_training(n_splits, num_epochs, device, num_labels=2):
    """ Stratified K-Fold maintains the same proportion of samples for each target """

    # Set deterministic behavior for reproducible results
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(RANDOM_SEED)

    # Initialize data handler with stratified K-fold configuration
    data_handler = DataHandler(DATASET_PATH, n_splits=n_splits)
    data_handler.load_data()
    data_handler.clean_data()

    # Store evaluation results from each fold
    fold_results = []

    # Iterate through each fold of the stratified cross-validation
    for fold, (train_texts, train_labels, test_texts, test_labels) in enumerate(data_handler.stratified_kfold_split_data()):
        print(f"Training on stratified fold {fold + 1}/{n_splits}...")

        # Create data loaders for current fold (no validation set in stratified K-fold)
        data_loader = PIIDataLoader(train_texts, train_labels, [], [], test_texts, test_labels)
        train_loader, _, test_loader = data_loader.get_dataloader()

        # Initialize a fresh model trainer for each fold (corretto il nome della classe)
        model_trainer = ClinicalT5_ModelTrainer(
            model_name=MODEL_NAME, 
            save_folder=SAVE_FOLDER, 
            device=device, 
            num_labels=num_labels,  # Aggiunto parametro num_labels
            num_epochs=num_epochs
        )

        # Train the model on current fold's training data
        model_trainer.train(train_loader, None) # No validation loader for stratified K-fold

        # Evaluate the model on current fold's test data
        test_results = model_trainer.evaluate(test_loader)
        
        # Prepare fold results for CSV
        fold_result = {
            'method': 'stratified_kfold',
            'fold': fold + 1,
            'epochs': num_epochs,
            'test_accuracy': test_results.get('accuracy', 0),
            'precision': test_results.get('precision', 0),
            'recall': test_results.get('recall', 0),
            'f1_score': test_results.get('f1_score', 0),
            'auc_roc': test_results.get('auc_roc', 0),
            'average_loss': test_results.get('loss', 0),
            'training_time_min': test_results.get('training_time_min', 0)
        }
        fold_results.append(fold_result)

        # Make sure the folder exists
        os.makedirs(SAVE_FOLDER, exist_ok=True)

        # Save the model and optimizer state with the folder prefix
        model_trainer.save_model(
            os.path.join(SAVE_FOLDER, MODEL_RES + f'_Strfold_{fold + 1}.pt'),
            os.path.join(SAVE_FOLDER, OPTIMIZER_RES + f'_Strfold_{fold + 1}.pt')
        )

    print(f"Stratified Cross-validation completed. Results collected for {len(fold_results)} folds.")

    # Calculate average results across all folds
    avg_results = calculate_average_results(fold_results)
    fold_results.append(avg_results)

    # Generate and display training metrics plots
    model_trainer.plot_metrics()

    # Save results to CSV
    save_results_to_csv(fold_results, f'stratified_kfold_{n_splits}splits')
    return fold_results

def calculate_average_results(fold_results):
    """Calculate average metrics across all folds"""
    if not fold_results:
        return {}
    
    # Calculate averages for numeric fields
    numeric_fields = ['test_accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'average_loss', 'training_time_min']
    avg_result = {
        'method': fold_results[0]['method'] + '_average',
        'fold': 'average',
        'epochs': fold_results[0]['epochs']
    }
    
    for field in numeric_fields:
        values = [result[field] for result in fold_results if isinstance(result[field], (int, float))]
        avg_result[field] = sum(values) / len(values) if values else 0
    
    return avg_result

def arg_commandline():
    """ Create an argument parser object """
    parser = argparse.ArgumentParser(description="ClinicalT5 model training with different validation strategies")

    # Add arguments/options
    parser.add_argument('-b', '--standard', action='store_true', help='enable standard train-validation-test split')
    parser.add_argument('-k', '--kfold', action='store_true', help='enable K-fold cross-validation')
    parser.add_argument('-s', '--strkfold', action='store_true', help='enable stratified k-fold cross-validation')
    # Add parameters for dataloader
    parser.add_argument('-n', '--num_splits', type=int, default=5, help='number of splits')
    parser.add_argument('-e', '--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--dataset', type=str, default='Dataset/', help='Path to the dataset file')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of classification labels')  # Nuovo parametro

    return parser.parse_args()


### Main
def main():
    args = arg_commandline()

    global DATASET_PATH
    DATASET_PATH = args.dataset


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Number of labels: {args.num_labels}")

    # Execute the appropriate training strategy based on user selection
    if args.standard:
        print("Starting standard train-validation-test split training...")
        results = training_test_validation(num_epochs=args.num_epochs, device=device, num_labels=args.num_labels)
    elif args.kfold:
        print("Starting K-Fold cross-validation training...")
        results = kfold_training(n_splits=args.num_splits, num_epochs=args.num_epochs, device=device, num_labels=args.num_labels)
    elif args.strkfold:
        print("Starting Stratified K-Fold cross-validation training...")
        results = stratified_kfold_training(n_splits=args.num_splits, num_epochs=args.num_epochs, device=device, num_labels=args.num_labels)
    else:
        script_path = os.path.abspath(__file__)
        script_name = os.path.basename(script_path)

        print("\nError: No training strategy selected!")
        print("Please choose one of the following options:")
        print("  -b, --standard    : Standard train-validation-test split")
        print("  -k, --kfold       : K-fold cross-validation")
        print("  -s, --strkfold    : Stratified K-fold cross-validation")
        print("\nExample usage:")
        print(f"  python3 {script_name} -s -n 3 -e 5 --dataset datasets/training_dataset.csv --num_labels 2")

if __name__ == "__main__":
    main()