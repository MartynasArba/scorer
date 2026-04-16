import torch
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from scorer.data.loaders import SleepSignals, SleepTraining, BufferedSleepDataset, SequenceSleepDataset

class TestSleepLoaders(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure for testing
        self.test_dir = Path(tempfile.mkdtemp())
        self.mouse_dir = self.test_dir / "mouse_001"
        self.mouse_dir.mkdir()
        
        self.n_samples = 30
        self.n_channels = 2 # EEG + EMG
        self.win_len = 1000
        
        # Create mock EEG/EMG data: [Channels, Samples, Win_Len]
        # Note: Loaders expect this specific internal format before permutation
        x_data = torch.randn(self.n_channels, self.n_samples, self.win_len)
        # Create mock labels: [1, Samples, 1]
        y_data = torch.cat([
            torch.full((1, 10, 1), 1), # Wake
            torch.full((1, 10, 1), 2), # NREM
            torch.full((1, 10, 1), 4)  # REM
        ], dim=1).long()
        
        torch.save(x_data, self.mouse_dir / "X_chunk1.pt")
        torch.save(y_data, self.mouse_dir / "y_chunk1.pt")
        
        self.metadata = {
            'ecog_channels': '0',
            'emg_channels': '1',
            'ylim': 'infer'
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_sleep_signals_basic_load(self):
        """Verify basic dataset loading and channel permutation."""
        ds = SleepSignals(
            data_path=str(self.mouse_dir), 
            score_path=str(self.mouse_dir), 
            metadata=self.metadata, 
            device='cpu'
        )
        self.assertEqual(len(ds), self.n_samples)
        sample, label = ds[0]
        # Should be [Channels, Win_Len]
        self.assertEqual(sample.shape, (self.n_channels, self.win_len))
        # Verify quantile-based ylims were calculated
        self.assertEqual(len(ds.channel_ylims), self.n_channels)

    def test_sleep_training_balancing(self):
        """Verify that undersampling correctly balances class distribution."""
        # Create dataset with imbalance (30 samples total)
        ds = SleepTraining(
            data_path=str(self.test_dir),
            n_files_to_pick=None,
            balance='undersample',
            exclude_labels=(0,),
            device='cpu'
        )
        
        labels = ds.all_labels.cpu().numpy()
        unique, counts = np.unique(labels, return_counts=True)
        # Since we had 10 of each class, and undersampling picks the min, 
        # all classes should now have the same count.
        self.assertTrue(all(c == counts[0] for c in counts), f"Classes not balanced: {counts}")

    def test_buffered_dataset_iteration(self):
        """Verify IterableDataset yields all samples across chunks."""
        ds = BufferedSleepDataset(
            data_path=str(self.test_dir),
            buffer_size=1,
            device='cpu'
        )
        
        samples_yielded = 0
        for x, y in ds:
            samples_yielded += 1
            if samples_yielded == 1:
                self.assertEqual(x.shape, (2, self.win_len))
        
        self.assertEqual(samples_yielded, self.n_samples)

    def test_sequence_dataset_sliding_window(self):
        """Verify sequence grouping for RNN context."""
        seq_len = 5
        ds = SequenceSleepDataset(
            data_path=str(self.test_dir),
            seq_len=seq_len,
            stride=1,
            device='cpu'
        )
        
        # Total valid starts = total samples - seq_len + 1
        expected_len = self.n_samples - seq_len + 1
        self.assertEqual(len(ds), expected_len)
        
        x_seq, y_seq = ds[0]
        # Shape: [Seq_Len, Channels, Win_Len]
        self.assertEqual(x_seq.shape, (seq_len, self.n_channels, self.win_len))
        self.assertEqual(y_seq.shape, (seq_len,))

if __name__ == "__main__":
    unittest.main()