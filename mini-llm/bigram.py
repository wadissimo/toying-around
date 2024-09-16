
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device", device)
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


#### Params 
batch_size = 64
block_size = 256
learning_iters = 5000
eval_interval = 500
lr = 3e-4
eval_iters = 100
embed_size=384
n_layers = 6
n_heads = 6
dropout = 0.2

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B,T,H
        q = self.query(x) #  B,T,H
        v = self.value(x) #  B,T,H
        wei = q @ k.transpose(-2,-1) * C**-0.5 # B,T,T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # lower triangular
        wei = F.softmax(wei, dim= -1) # B,T,T
        wei = self.dropout(wei)
        out = wei @ v # B,T,H
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads =  nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# Feed Forward N->N
class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.ReLU(),
            nn.Linear(4*embed_size, embed_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.ff(x)
    

class Block(nn.Module):
    def __init__(self, embed_size, n_heads):
        super().__init__()
        self.att_head = MultiHeadAttention(n_heads, embed_size//n_heads) ##Head(embed_size)
        self.ffwd = FeedForward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        x = x + self.att_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding_table = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[Block(embed_size, n_heads) for _ in range(n_layers)])
        self.ln1 = nn.LayerNorm(embed_size)
        self.lin_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx) # B,T,C . C=embed_size
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln1(x)
        logits = self.lin_head(x)

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
            idx_crop = idx[:, -block_size:]
            logits, loss = self(idx_crop)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next), dim=1)
        return idx



@torch.no_grad()
def eval_loss(model):
    res = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        res[split] = losses.mean()
    model.train()
    return res
        
model = BigramLanguageModel()
model = model.to(device)
data = data.to(device)
train_data = train_data.to(device)
val_data = val_data.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("Start training")
for steps in range(learning_iters):

    xb, yb = get_batch('train')
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % eval_interval == 0:
        losses = eval_loss(model)
        print(f"step: {steps}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

#print("final Loss:", loss.item())
print("Output:")
print(decode(model.generate(torch.zeros((1,1),dtype=torch.long).to(device), max_new_tokens=5000)[0].tolist()))