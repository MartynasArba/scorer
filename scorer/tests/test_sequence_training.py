import torch
import unittest
import tempfile
import shutil
from pathlib import Path
from scorer.models.sequence_training import train_sequence_model
from scorer.models.sleep_cnn import SCDSSleepCNN

class MockSequenceDataset(torch.utils.data.Dataset):
    """Minimal dataset to simulate SequenceSleepDataset."""
    def __init__(self, seq_len=10, n_samples=20, channels=1, win_len=1000, device='cpu'):
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.device = device
        # [Batch, Seq, Chan, Time] - Single channel input for SCDS model
        self.data = torch.randn(n_samples, seq_len, channels, win_len)
        # Satisfy train_sequence_model expectation of all_samples attribute
        self.all_samples = self.data
        self.labels = torch.randint(0, 3, (n_samples, seq_len))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TestSequenceTraining(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.input_channels = 1 # Single cortical channel
        self.win_len = 1000

        # Create a dummy pretrained encoder
        self.encoder_path = self.test_dir / "dummy_encoder.pt"
        encoder = SCDSSleepCNN(num_classes=3)
        # Trigger LazyLinear with a dummy pass, matching the expected input shape for the encoder
        encoder(torch.randn(1, self.input_channels, self.win_len))
        torch.save(encoder.state_dict(), self.encoder_path)
        
        self.dataset = MockSequenceDataset(channels=self.input_channels, win_len=self.win_len, device=self.device)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_training_loop_sanity(self):
        """Verify that the training function runs and produces a model."""
        # Run for just 1 epoch to verify pipeline
        model = train_sequence_model(
            self.dataset, 
            self.dataset, # Use same dataset for validation in sanity check
            encoder_path=str(self.encoder_path), 
            epochs=1, 
            batch_size=2
        )
        
        self.assertIsNotNone(model)
        self.assertTrue(Path(self.test_dir / "3state_SCDS_GRU_weights.pt").exists())

    def test_gradient_isolation(self):
        """Confirm that only the RNN and Classifier heads have gradients."""
        model = train_sequence_model(
            self.dataset, 
            self.dataset,
            encoder_path=str(self.encoder_path), 
            epochs=1, 
            batch_size=2
        )
        
        # Check encoder (should have no grads)
        for param in model.encoder.parameters():
            self.assertTrue(param.grad is None or torch.all(param.grad == 0))
            
        # Check RNN (should have grads if optimized)
        # Note: In a 1-epoch test, we just check requires_grad
        self.assertTrue(any(p.requires_grad for p in model.rnn.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))

    def test_output_mapping(self):
        """Verify logits are correctly transposed for CrossEntropy [B, C, S]."""
        model = train_sequence_model(self.dataset, str(self.encoder_path), epochs=0, batch_size=2) # batch_size > 0 needed for model creation
        model = train_sequence_model(
            self.dataset, 
            self.dataset, 
            str(self.encoder_path), 
            epochs=0, 
            batch_size=2
        ) # batch_size > 0 needed for model creation
        dummy_input = torch.randn(1, self.dataset.seq_len, self.input_channels, self.win_len).to(self.device)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 3, 10))

if __name__ == "__main__":
    unittest.main()