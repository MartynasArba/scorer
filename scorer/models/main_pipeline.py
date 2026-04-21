import torch
import logging
import datetime
import sys
import traceback
from pathlib import Path
import os

from scorer.data.loaders import BufferedSleepDataset, SequenceSleepDataset
from scorer.models.pretraining import train_unsupervised, train_supcon
from scorer.models.sequence_training import train_sequence_model
from scorer.models.sleep_cnn import SCDSSleepCNN
from scorer.models.contrastive_embedder import SupConSleepCNN

def setup_global_logger(save_dir: Path):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = save_dir / f"full_pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("PipelineOrchestrator")

def log_system_info(logger):
    logger.info("="*50)
    logger.info("SYSTEM DIAGNOSTICS")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info("="*50)

def run_full_pipeline():
    # --- CONFIGURATION ---
    CONFIG = {
        "paths": {
            "unlabeled_data": r"C:\Users\marty\Desktop\train_sets\unlabeled",
            "labeled_data": r"C:\Users\marty\Desktop\train_sets\labeled",
            "weights_dir": Path(r"C:\Users\marty\Projects\scorer\scorer\models\weights"),
            "val_data": r"C:\Users\marty\Desktop\train_sets\val"
        },
        "pretrain": {
            "batch_size": 1024,
            "simclr_epochs": 50,
            "supcon_epochs": 50,
            "n_files_buffer": 100,
        },
        "sequence": {
            "batch_size": 128,
            "epochs": 20,
            "seq_len": 10,
        },
        "metadata": {
            'ecog_channels': '0', 
            'emg_channels': '1', 
            'sample_rate': '250', 
            'ylim': 'standard'
        }
    }

    CONFIG["paths"]["weights_dir"].mkdir(parents=True, exist_ok=True)
    logger = setup_global_logger(CONFIG["paths"]["weights_dir"])
    log_system_info(logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.datetime.now().strftime('%Y%m%d')
    
    try:
        # --- STEP 1: SIMCLR PRETRAINING ---
        logger.info("STAGE 1: Starting Unsupervised SimCLR Pretraining")
        
        unsup_dataset = BufferedSleepDataset(
            data_path=CONFIG["paths"]["unlabeled_data"],
            n_files_to_pick=None,
            buffer_size=CONFIG["pretrain"]["n_files_buffer"],
            metadata=CONFIG["metadata"],
            normalize= True,
            merge_nrem=True,
            device='cpu' # Load chunks to CPU RAM then move to GPU in loop
        )

        base_cnn = SCDSSleepCNN(num_classes=3).to(device)
        contrastive_model = SupConSleepCNN(base_cnn).to(device)

        pretrained_cnn, _, _ = train_unsupervised(
            contrastive_model, 
            unsup_dataset, 
            logger, 
            save_dir=CONFIG["paths"]["weights_dir"], 
            epochs=CONFIG["pretrain"]["simclr_epochs"], 
            batch_size=CONFIG["pretrain"]["batch_size"]
        )
        
        simclr_weights_path = CONFIG["paths"]["weights_dir"] / f"SimCLR_weights_{timestamp}.pt"
        torch.save(pretrained_cnn.state_dict(), simclr_weights_path)
        logger.info(f"SimCLR stage complete. Weights saved to {simclr_weights_path}")

        del unsup_dataset # Clear RAM before next stage

        # --- STEP 2: SUPCON PRETRAINING ---
        logger.info("STAGE 2: Starting Supervised SupCon Pretraining")
        
        sup_dataset = BufferedSleepDataset(
            data_path=CONFIG["paths"]["labeled_data"],
            n_files_to_pick=None,
            buffer_size=CONFIG["pretrain"]["n_files_buffer"],
            metadata=CONFIG["metadata"],
            normalize= True,
            balance='undersample',
            merge_nrem=True,
            device='cpu'
        )

        final_encoder, _ = train_supcon(
            contrastive_model, 
            sup_dataset, 
            logger, 
            save_dir=CONFIG["paths"]["weights_dir"], 
            epochs=CONFIG["pretrain"]["supcon_epochs"], 
            batch_size=CONFIG["pretrain"]["batch_size"]
        )

        supcon_weights_path = CONFIG["paths"]["weights_dir"] / f"SupCon_final_{timestamp}.pt"
        torch.save(final_encoder.state_dict(), supcon_weights_path)
        logger.info(f"SupCon stage complete. Encoder saved to {supcon_weights_path}")

        del sup_dataset

        # --- STEP 3: SEQUENCE TRAINING ---
        logger.info("STAGE 3: Starting Sequence Model Training (GRU)")
        
        seq_dataset = SequenceSleepDataset(
            data_path=CONFIG["paths"]["labeled_data"],
            seq_len=CONFIG["sequence"]["seq_len"],
            normalize= True,
            stride=1,
            device=device,
            merge_nrem=True,
            augment=True
        )
        
        seq_val_dataset = SequenceSleepDataset(
            data_path=CONFIG["paths"]["val_data"],
            seq_len=CONFIG["sequence"]["seq_len"],
            normalize= True,
            stride=1,
            device=device,
            merge_nrem=True,
            augment=False
        )

        train_sequence_model(
            seq_dataset, 
            seq_val_dataset,
            str(supcon_weights_path), 
            logger=logger, 
            epochs=CONFIG["sequence"]["epochs"], 
            batch_size=CONFIG["sequence"]["batch_size"]
        )

        logger.info("PIPELINE SUCCESS: All stages completed successfully.")

    except Exception as e:
        logger.error("PIPELINE CRITICAL FAILURE")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_full_pipeline()
