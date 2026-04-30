import torch
import torch.nn as nn
import unittest
import tempfile
import shutil
from pathlib import Path
import logging
import sys
import os

# Ensure project root is in path for imports
sys.path.append(os.getcwd())

from scorer.data.loaders import BufferedSleepDataset, SequenceSleepDataset
from scorer.models.sleep_cnn import SCDSSleepCNN
from scorer.models.contrastive_embedder import SupConSleepCNN
from scorer.models.pretraining import train_unsupervised, train_supcon
from scorer.models.adversarial_training import train_adversarial_domain
from scorer.models.sequence_training import train_sequence_model

class TestSleepScorerPipeline(unittest.TestCase):
    """
    Integration tests to verify the functional integrity of the sleep scoring pipeline stages.
    """
    @classmethod
    def setUpClass(cls):
        # Create temporary environment
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.weights_dir = cls.test_dir / "weights"
        cls.weights_dir.mkdir()
        
        # Create dummy data folders for each stage
        cls.labeled_path = cls.test_dir / "labeled" / "mouse1"
        cls.unlabeled_path = cls.test_dir / "unlabeled" / "mouse2"
        cls.val_path = cls.test_dir / "val" / "mouse3"
        for p in [cls.labeled_path, cls.unlabeled_path, cls.val_path]:
            p.mkdir(parents=True)
            
        cls.win_len = 1000
        cls.channels = 2
        cls.n_samples = 4
        
        # Create dummy .pt files representing the expected windowed structure
        def save_dummy(path, prefix):
            # Shape: [Channels, Samples, Win_Len]
            X = torch.randn(cls.channels, cls.n_samples, cls.win_len)
            # Shape: [1, Samples, Win_Len]
            y = torch.randint(1, 5, (1, cls.n_samples, cls.win_len))
            torch.save(X, path / f"X_{prefix}.pt")
            torch.save(y, path / f"y_{prefix}.pt")
            
        save_dummy(cls.labeled_path, "chunk0")
        save_dummy(cls.unlabeled_path, "chunk0")
        save_dummy(cls.val_path, "chunk0")
        
        cls.meta = {
            'ecog_channels': '0', 
            'emg_channels': '1', 
            'sample_rate': '250', 
            'ylim': 'standard'
        }
        
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Suppress logging to keep test output clean
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_01_buffered_dataset(self):
        """Verify BufferedSleepDataset correctly loads and remaps mock data."""
        ds = BufferedSleepDataset(
            data_path=[str(self.labeled_path.parent)],
            n_files_to_pick=None,
            buffer_size=1,
            metadata=self.meta,
            merge_nrem=True,
            device='cpu'
        )
        self.assertEqual(len(ds), self.n_samples)
        x, y = next(iter(ds))
        # BufferedSleepDataset returns (Channels, Win_Len) per sample
        self.assertEqual(x.shape, (self.channels, self.win_len))
        # Verify remapping (1,2,4 -> 0,1,2)
        self.assertTrue(y.item() in [0, 1, 2])

    def test_02_lazy_init_and_forward(self):
        """Verify SCDSSleepCNN lazy initialization and output shapes."""
        model = SCDSSleepCNN(num_classes=3).to(self.device)
        # Use batch size > 1 to satisfy BatchNorm
        dummy = torch.randn(2, 1, self.win_len).to(self.device)
        out = model(dummy)
        # Output should be embedding dimension (256 time + 256 freq)
        self.assertEqual(out.shape, (2, 512))

    def test_03_simclr_loop_sanity(self):
        """Test one epoch of SimCLR pretraining."""
        base_cnn = SCDSSleepCNN(num_classes=3).to(self.device)
        model = SupConSleepCNN(base_cnn).to(self.device)
        # Dummy pass is REQUIRED before SimCLR loop creates optimizer
        # Batch size 2 avoids BatchNorm issues
        model(torch.randn(2, 1, self.win_len).to(self.device))
        
        ds = BufferedSleepDataset(
            data_path=[str(self.unlabeled_path.parent)], 
            metadata=self.meta,
            merge_nrem=True
        )
        logger = logging.getLogger("test")
        
        train_unsupervised(model, ds, logger, save_dir=self.weights_dir, epochs=1, batch_size=2)

    def test_04_supcon_loop_sanity(self):
        """Test one epoch of SupCon pretraining."""
        base_cnn = SCDSSleepCNN(num_classes=3).to(self.device)
        model = SupConSleepCNN(base_cnn).to(self.device)
        model(torch.randn(2, 1, self.win_len).to(self.device))
        
        ds = BufferedSleepDataset(
            data_path=[str(self.labeled_path.parent)], 
            metadata=self.meta, 
            balance='undersample',
            merge_nrem=True
        )
        logger = logging.getLogger("test")
        
        train_supcon(model, ds, logger, save_dir=self.weights_dir, epochs=1, batch_size=2)

    def test_05_adversarial_loop_sanity(self):
        """Test one epoch of adversarial alignment."""
        encoder = SCDSSleepCNN(num_classes=3).to(self.device)
        encoder(torch.randn(2, 1, self.win_len).to(self.device))
        
        labeled_ds = BufferedSleepDataset(
            data_path=[str(self.labeled_path.parent)], 
            metadata=self.meta,
            merge_nrem=True
        )
        ood_ds = BufferedSleepDataset(
            data_path=[str(self.unlabeled_path.parent)], 
            metadata=self.meta,
            merge_nrem=True
        )
        
        from torch.utils.data import DataLoader
        l_loader = DataLoader(labeled_ds, batch_size=2)
        o_loader = DataLoader(ood_ds, batch_size=2)
        
        opt = torch.optim.Adam(encoder.parameters(), lr=1e-4)
        logger = logging.getLogger("test")
        
        train_adversarial_domain(encoder, l_loader, o_loader, opt, logger, epochs=1)

    def test_06_sequence_loop_sanity(self):
        """Test one epoch of sequence training."""
        encoder = SCDSSleepCNN(num_classes=3).to(self.device)
        encoder(torch.randn(2, 1, self.win_len).to(self.device))
        weights_path = self.weights_dir / "test_enc.pt"
        torch.save(encoder.state_dict(), weights_path)
        
        seq_ds = SequenceSleepDataset(
            data_path=str(self.labeled_path.parent), 
            seq_len=2, 
            stride=1, 
            device=self.device,
            merge_nrem=True
        )
        val_ds = SequenceSleepDataset(
            data_path=str(self.val_path.parent), 
            seq_len=2, 
            stride=2, 
            device=self.device,
            merge_nrem=True
        )
        
        logger = logging.getLogger("test")
        train_sequence_model(seq_ds, val_ds, str(weights_path), logger=logger, epochs=1, batch_size=2)

if __name__ == "__main__":
    unittest.main()