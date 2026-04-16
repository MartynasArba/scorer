import torch
import torch.nn as nn
import unittest
from scorer.models.sequence_model import ContextAwareSleepScorer, FocalLoss
from scorer.models.sleep_cnn import SCDSSleepCNN

class TestSequenceModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.seq_len = 10
        self.channels = 1
        self.win_len = 1000
        self.num_classes = 3
        
        # Initialize a base encoder
        self.base_encoder = SCDSSleepCNN(num_classes=self.num_classes).to(self.device)
        # Initialize sequence model
        self.model = ContextAwareSleepScorer(
            self.base_encoder, 
            embedding_dim=512, 
            hidden_dim=32, 
            num_classes=self.num_classes, 
            num_layers=1
        ).to(self.device)

    def test_forward_shape(self):
        """Verify input [B, S, C, T] results in [B, Class, S] for CrossEntropy compatibility."""
        # Shape: [Batch, Sequence, Channels, Time]
        x = torch.randn(self.batch_size, self.seq_len, self.channels, self.win_len).to(self.device)
        logits = self.model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes, self.seq_len))

    def test_encoder_frozen(self):
        """Ensure the CNN parameters do not accumulate gradients."""
        x = torch.randn(self.batch_size, self.seq_len, self.channels, self.win_len).to(self.device)
        logits = self.model(x)
        loss = logits.sum()
        loss.backward()
        
        for name, param in self.model.encoder.named_parameters():
            self.assertIsNone(param.grad, f"Parameter {name} in encoder should be frozen.")

    def test_temporal_dependency(self):
        """
        Verify that changing the first item in a sequence affects subsequent outputs.
        This confirms the GRU is actually passing hidden states across the sequence.
        """
        self.model.eval()
        x = torch.randn(1, self.seq_len, self.channels, self.win_len).to(self.device)
        
        with torch.no_grad():
            out1 = self.model(x)
            
            # Perturb only the first window in the sequence
            x_perturbed = x.clone()
            x_perturbed[0, 0, :, :] += 10.0
            out2 = self.model(x_perturbed)
            
        # The output at index 5 should change because of the Bi-GRU context
        diff = torch.abs(out1[0, :, 5] - out2[0, :, 5]).sum()
        self.assertGreater(diff.item(), 1e-5, "GRU failed to propagate temporal context.")

    def test_focal_loss_scaling(self):
        """Verify FocalLoss correctly penalizes easy examples less than standard CE."""
        criterion_focal = FocalLoss(gamma=2.0)
        
        # Target is class 1
        target = torch.tensor([1]).to(self.device)
        
        # Case 1: High confidence (Easy example)
        logits_easy = torch.tensor([[0.0, 10.0, 0.0]]).to(self.device)
        # Case 2: Low confidence (Hard example)
        logits_hard = torch.tensor([[0.0, 0.1, 0.0]]).to(self.device)
        
        loss_easy = criterion_focal(logits_easy, target)
        loss_hard = criterion_focal(logits_hard, target)
        
        # With gamma=2, the easy loss should be vanishingly small compared to hard loss
        self.assertLess(loss_easy, loss_hard)
        
        # Check that loss is positive
        self.assertGreater(loss_easy.item(), 0)

    def test_identity_mapping_safety(self):
        """Ensure the encoder's output dimension matches the RNN's expected input."""
        self.assertEqual(self.model.rnn.input_size, 512, "RNN input size mismatch with SCDSSleepCNN output.")

if __name__ == "__main__":
    unittest.main()