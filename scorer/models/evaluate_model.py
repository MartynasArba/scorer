import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score, f1_score
from pathlib import Path
import logging
from datetime import datetime

from scorer.data.loaders import SequenceSleepDataset
from scorer.models.scoring import load_trained_sequence_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ModelEvaluator")

def evaluate_model_on_unseen(data_path: str, model_path: str, device: str = 'cuda', batch_size: int = 128):
    """
    Evaluates a trained sequence model on labeled unseen data.
    
    Args:
        data_path: Path to the folder containing labeled .pt files for evaluation.
        model_path: Path to the .pt file containing the trained model weights.
        device: 'cuda' or 'cpu'.
        batch_size: Number of sequences per batch.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # load dataset
    # use SequenceSleepDataset to handle [Batch, Seq, Chan, Time] format
    logger.info(f"Loading unseen dataset from {data_path}...")
    try:
        dataset = SequenceSleepDataset(
            data_path=data_path,
            seq_len=10, 
            stride=1,
            device=device,
            normalize=True, # Ensure unseen data is normalized
            merge_nrem=True,
            augment=False
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # load model
    # load_trained_sequence_model handles LazyLinear init and loading weights.
    logger.info(f"Loading model from {model_path}...")
    try:
        model = load_trained_sequence_model(model_path, device=device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
        
    model.eval()

    all_preds = []
    all_labels = []

    logger.info(f"Running inference on {len(dataset)} sequences...")
    with torch.no_grad():
        for samples, labels in loader:
            samples = samples.to(device)
            # model output shape: [Batch, Classes, Seq_Len]
            print(f"eval batch Max: {samples.max().item():.4f}, Min: {samples.min().item():.4f}, Mean: {samples.mean().item():.4f}")
            outputs = model(samples)
            
            # get predictions[Batch, Seq_Len]
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())

    if not all_preds:
        logger.error("No data found to evaluate.")
        return

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # calculate metrics
    acc = (y_true == y_pred).mean()
    f1_macro = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)

    print("\n" + "="*50)
    print("                MODEL PERFORMANCE                ")
    print("="*50)
    print(f"Global Accuracy:   {acc:.4f}")
    print(f"Macro F1-Score:    {f1_macro:.4f}")
    print(f"Cohen's Kappa:     {kappa:.4f}")
    print("-" * 50)
    
    target_names = ['Wake', 'NREM', 'REM']
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("="*50)

    # visualize confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues', ax=ax, values_format='.2f')
    
    plt.title(f"Confusion Matrix (Normalized)\n{Path(model_path).name}")
    
    save_fig_path = Path(model_path).parent / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(save_fig_path)
    logger.info(f"Confusion matrix saved to: {save_fig_path}")
    plt.show()

if __name__ == "__main__":
    # Configuration
    # print("F channel")
    # VAL_DATA = r'C:\Users\marty\Desktop\train_sets\final_test\F'
    # MODEL_WEIGHTS = r'C:\Users\marty\Projects\scorer\scorer\models\weights\3state_SCDS_GRU_weights.pt'
    # # evaluate_model_on_unseen(data_path: str, model_path: str, device: str = 'cuda', batch_size: int = 128):
    # evaluate_model_on_unseen(VAL_DATA, MODEL_WEIGHTS)
    # print('P channel')
    VAL_DATA = r'C:\Users\marty\Desktop\train_sets\final_test\P'
    MODEL_WEIGHTS = r'C:\Users\marty\Projects\scorer\scorer\models\weights\3state_SCDS_GRU_weights.pt'
    # evaluate_model_on_unseen(data_path: str, model_path: str, device: str = 'cuda', batch_size: int = 128):
    evaluate_model_on_unseen(VAL_DATA, MODEL_WEIGHTS)