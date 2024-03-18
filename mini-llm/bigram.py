
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_file = "data/input.txt"


#read input text for training and validation
with open(data_file) as f:
    text = f.read()

# prepare tokens
vocab = set()
for c in text:
    vocab.add(c)
vocab_size = len(vocab)
stoi = {c:i for i, c in enumerate(vocab)}
itos = {i:c for i, c in enumerate(vocab)}
def encode(s): return [stoi[c] for c in s]
def decode(a): return ''.join([itos[i] for i in a])

# prepare train and val datasets
data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]

batch_size = 4
block_size = 8

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss =  F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next), dim=1)
        return idx
            
        
m = BigramLanguageModel(vocab_size)
m = m.to(device)
data = data.to(device)
train_data = train_data.to(device)
val_data = val_data.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size=32

for steps in range(10000):
    xb, yb = get_batch('train')
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Loss:", loss.item())
print("Output:")
print(decode(m.generate(torch.zeros((1,1),dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))