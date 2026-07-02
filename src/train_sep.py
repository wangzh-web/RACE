"""
(SEP)"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

class SemanticEntropyProbe(nn.Module):
    
    def __init__(self, input_dim=3072, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_sep(epochs=100, batch_size=32, lr=1e-3):
    
    print("Loading...")
    data = torch.load('data/sep/training_data.pt')
    X = data['hidden_states'].float()
    y = data['semantic_entropies'].float()
    
    print(f"X={X.shape}, y={y.shape}")
    
    n = len(X)
    train_size = int(0.8 * n)
    
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(X[train_size:], y[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = SemanticEntropyProbe(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    print(f"\n({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models/sep', exist_ok=True)
            torch.save(model.state_dict(), 'models/sep/best_sep.pt')
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss/len(train_loader):.4f}, Val={val_loss:.4f}")
    
    print(f"\n✅ done!")
    print(f"MSE: {best_val_loss:.4f}")
    print(f"models/sep/best_sep.pt")

if __name__ == "__main__":
    train_sep()
