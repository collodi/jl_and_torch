import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from data import read_bods, read_caps

data_dir = './data/rand_tiny'

X = torch.stack(read_bods(data_dir))
Y = torch.stack(read_caps(data_dir))

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = DataLoader(list(zip(X, Y)), batch_size=5, shuffle=False)

class Dense(nn.Module):
	def __init__(self):
		super().__init__()
		self.dense = nn.Sequential(
			nn.Linear(8, 4),
			nn.ReLU(),
			nn.Linear(4, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = torch.flatten(x, start_dim=1)
		return self.dense(x)

def train(m, dev, loader, opt):
	m.train()
	for i, (x, y) in enumerate(loader):
		x, y = x.to(dev), y.to(dev)

		opt.zero_grad()
		y_hat = m(x)

		loss = F.mse_loss(y_hat, y)
		loss.backward()
		opt.step()

def train_loss(m, dev):
	m.eval()
	with torch.no_grad():
		Y_hat = m(X.to(dev)).view(-1)
		return F.l1_loss(Y_hat, Y)

m = Dense().to(dev)
opt = optim.Adam(m.parameters(), lr=1e-2)

nepochs = 500
for epoch in range(nepochs):
	train(m, dev, loader, opt)

	if epoch % 100 == 0:
		loss = train_loss(m, dev)
		print(f'epoch {epoch}, mae {loss:.6f}')
