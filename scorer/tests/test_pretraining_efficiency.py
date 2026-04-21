import torch
import unittest
import numpy as np
from scorer.models.sleep_cnn import SCDSSleepCNN
from scorer.models.contrastive_embedder import SupConSleepCNN, SupConLoss, SimCLRLoss
from scorer.models.pretraining import batch_augment

class TestPretrainingEfficiency(unittest.TestCase):
    """
    Unit tests to verify pretraining logic and gauge if extended training is likely to yield gains.
    """
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.batch_size = 32
        cls.win_len = 1000
        cls.input_channels = 1

    def setUp(self):
        self.base_model = SCDSSleepCNN(num_classes=3).to(self.device)
        
        # Initialize LazyLinear layers with a dummy pass before the model is wrapped/optimizer is created
        with torch.no_grad():
            dummy = torch.randn(1, self.input_channels, self.win_len).to(self.device)
            self.base_model(dummy)

        self.model = SupConSleepCNN(self.base_model).to(self.device)
        # Synthetic data simulating EEG epochs
        self.samples = torch.randn(self.batch_size, self.input_channels, self.win_len).to(self.device)
        self.labels = torch.randint(0, 3, (self.batch_size,)).to(self.device)

    def test_augmentation_validity(self):
        """Verify that batch_augment generates valid, non-identical views."""
        v1 = batch_augment(self.samples)
        v2 = batch_augment(self.samples)
        self.assertEqual(v1.shape, self.samples.shape)
        # Check that views are different (stochasticity check)
        self.assertFalse(torch.allclose(v1, v2), "Augmentations must be stochastic.")

    def test_overfitting_sanity(self):
        """
        Tests if the model can minimize loss on a static batch.
        If the model cannot overfit a small batch quickly, long pretraining is likely ineffective 
        due to architectural bottlenecks or hyperparameter issues.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = SupConLoss(temperature=0.07)
        
        initial_loss = None
        for i in range(10):
            optimizer.zero_grad()
            v1 = batch_augment(self.samples)
            v2 = batch_augment(self.samples)
            features = self.model(torch.cat([v1, v2], dim=0))
            combined_labels = torch.cat([self.labels, self.labels], dim=0)
            loss = criterion(features, combined_labels)
            loss.backward()
            optimizer.step()
            
            if i == 0: initial_loss = loss.item()
            
        self.assertLess(loss.item(), initial_loss, "Loss failed to decrease on a static batch subset.")

    def test_embedding_discrimination(self):
        """
        Measures if the model can develop a positive similarity spread.
        Runs a few optimization steps first to ensure the model moves away from 
        random initialization and lazy layers are initialized.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = SupConLoss(temperature=0.07)
        
        # Short training to break symmetry and initialize lazy layers
        for _ in range(5):
            optimizer.zero_grad()
            v1 = batch_augment(self.samples)
            v2 = batch_augment(self.samples)
            features = self.model(torch.cat([v1, v2], dim=0))
            combined_labels = torch.cat([self.labels, self.labels], dim=0)
            loss = criterion(features, combined_labels)
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            features = self.model(self.samples)
            sim_matrix = torch.matmul(features, features.T)
            
            # Masks for positive (same class) and negative (different class) pairs
            label_mask = torch.eq(self.labels.unsqueeze(1), self.labels.unsqueeze(0)).float()
            pos_mask = label_mask - torch.eye(self.batch_size, device=self.device)
            neg_mask = 1.0 - label_mask
            
            pos_sim = (sim_matrix * pos_mask).sum() / pos_mask.sum().clamp(min=1)
            neg_sim = (sim_matrix * neg_mask).sum() / neg_mask.sum().clamp(min=1)
            
            spread = (pos_sim - neg_sim).item()
            print(f"\n[Metric] Embedding Similarity Spread: {spread:.4f}")
            self.assertGreater(pos_sim, neg_sim, "Positives should be more similar than negatives.")

if __name__ == "__main__":
    unittest.main()
