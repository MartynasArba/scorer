import torch
import logging
import datetime
import sys
import traceback
from pathlib import Path
import os

from torch.utils.data import DataLoader
import torch.optim as optim

from scorer.data.loaders import BufferedSleepDataset, SequenceSleepDataset
from scorer.models.pretraining import train_unsupervised, train_supcon
from scorer.models.sequence_training import train_sequence_model
from scorer.models.sleep_cnn import SCDSSleepCNN
from scorer.models.contrastive_embedder import SupConSleepCNN
from scorer.models.adversarial_training import train_adversarial_domain

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
            "supcon_epochs": 100,
            "n_files_buffer": 100,
        },
        "adversarial": {
            "batch_size": 1024,
            "epochs": 10,
            "n_files_buffer": 100,
            "win_len": 1000,
        },
        "sequence": {
            "batch_size": 128,
            "epochs": 20,
            "seq_len": 20,
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
            data_path=[CONFIG["paths"]["labeled_data"], CONFIG["paths"]["unlabeled_data"]],
            n_files_to_pick=None,
            buffer_size=CONFIG["pretrain"]["n_files_buffer"],
            metadata=CONFIG["metadata"],
            normalize= True,
            merge_nrem=True,
            device='cpu' # Load chunks to CPU RAM then move to GPU in loop
        )

        base_cnn = SCDSSleepCNN(num_classes=3).to(device)
        contrastive_model = SupConSleepCNN(base_cnn).to(device)

        # Initialize LazyLinear layers before optimizer creation in training functions
        logger.info("Initializing Lazy layers with dummy pass...")
        dummy_in = torch.randn(1, 1, CONFIG["adversarial"].get("win_len", 1000)).to(device)
        contrastive_model(dummy_in)

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

        encoder_save_path = CONFIG["paths"]["weights_dir"] / f"SupCon_final_{timestamp}.pt"
        torch.save(final_encoder.state_dict(), encoder_save_path)
        logger.info(f"SupCon stage complete. Encoder saved to {encoder_save_path}")

        del sup_dataset
        
        # --- STEP 2.5: OOD ALIGNMENT, optional ---
        if 'adversarial' in CONFIG.keys():
                
            print('adversarial training started')
            
            labeled_dataset = BufferedSleepDataset(
                    data_path=CONFIG["paths"]["labeled_data"],
                    n_files_to_pick=None,
                    buffer_size=CONFIG["adversarial"]["n_files_buffer"],
                    metadata=CONFIG["metadata"],
                    normalize= True,
                    merge_nrem=True,
                    device='cpu'
                )
            
            ood_dataset = BufferedSleepDataset(
                    data_path=CONFIG["paths"]["unlabeled_data"],
                    n_files_to_pick=None,
                    buffer_size=CONFIG["adversarial"]["n_files_buffer"],
                    metadata=CONFIG["metadata"],
                    normalize= True,
                    merge_nrem=True,
                    device='cpu'
                )
            
            good_loader = DataLoader(labeled_dataset, batch_size= CONFIG["adversarial"]["batch_size"], shuffle = False, drop_last=True)    #shuffle is handled by the dataset itself
            ood_loader = DataLoader(ood_dataset, batch_size= CONFIG["adversarial"]["batch_size"], shuffle = False, drop_last=True)
                    
            encoder = final_encoder
            
            adversarial_optimizer = optim.Adam([
                {'params': encoder.time_conv.parameters(), 'lr': 2e-6}, #got by tuning
                {'params': encoder.freq_conv.parameters(), 'lr': 2e-6},
                {'params': encoder.freq_flatten.parameters(), 'lr': 1e-5},
                {'params': encoder.domain_classifier.parameters(), 'lr': 1e-3},#, 'weight_decay': 1e-3
                {'params': encoder.state_classifier.parameters(), 'lr': 1e-3}# fast learning for new heads
            ])
            
            encoder = train_adversarial_domain(encoder, good_loader, ood_loader, adversarial_optimizer, logger, epochs = CONFIG["adversarial"]["epochs"])
            
            encoder_save_path = CONFIG["paths"]["weights_dir"] / f"adversarial_adjusted_encoder{timestamp}.pt"
            torch.save(encoder.state_dict(), encoder_save_path)
            logger.info(f"Adversarial training stage complete. Encoder saved to {encoder_save_path}")
            
            del labeled_dataset, ood_dataset

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
            stride=CONFIG["sequence"]["seq_len"],
            device=device,
            merge_nrem=True,
            augment=False
        )

        train_sequence_model(
            seq_dataset, 
            seq_val_dataset,
            str(encoder_save_path), 
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
