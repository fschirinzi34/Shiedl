import torch
import pandas as pd
import os
from datetime import datetime
from custom_libs_clinical_t5_base.data_handler import DataHandler
from custom_libs_clinical_t5_base.pii_data_loader import PIIDataLoader
from custom_libs_clinical_t5_base.ClinicalT5_ModelTrainer import ClinicalT5_ModelTrainer


### Constants
RANDOM_SEED=123 # Fixed seed for reproducible results
WEIGHT_MODEL_PATH= r'C:\Users\Gabriele\Desktop\NER\Clinical_T5_TESTING\Base\ClinicalT5-base_model_tte.pt' # Model from Hugging Face
MODEL_NAME='hossboll/clinical-t5' # Model from Hugging Face
SAVE_FOLDER='results_train' # Path where to save the data

### Functions
def save_results_to_csv(results):
    """Simple version that saves directly to current directory"""

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"testing_results_{timestamp}.csv"

        # Create results directory if it doesn't exist
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        filepath = os.path.join(SAVE_FOLDER, filename)

        print(f"Saving results to: {filepath}")
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
        
        if os.path.exists(filepath):
            print(f"File created successfully: {filepath}")
            return filepath
        else:
            print("File was not created!")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def evaluation(device, num_labels=2):

    # Set deterministic behavior for reproducible results
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(RANDOM_SEED)

    # Initialize data handler and load the test dataset
    data_handler = DataHandler(DATASET_PATH, train_size=0, valid_size=0, test_size=1)
    data_handler.load_data()
    data_handler.clean_data()

    # Split data into training, validation, and test sets
    test_texts, test_labels = data_handler.get_test_data()

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

### Main
def main():

    global DATASET_PATH
    DATASET_PATH = "test_dataset.csv"

    num_labels = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using model: {WEIGHT_MODEL_PATH}")
    print(f"Number of labels: {num_labels}")

    print()
    print("Starting evaluation...")
    results = evaluation(device=device, num_labels=num_labels)
    print(results)

if __name__ == "__main__":
    main()
