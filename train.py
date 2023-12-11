from model import TuckerFormer, TuckerFormerConfig
import torch
from transformers.optimization import Adafactor
from tqdm import tqdm

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 512  # what is the maximum context length for predictions?
max_iters = 1500
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 64
n_head = 8
n_layer = 12
dropout = 0.0
bias = False
distmult = False
# ------------
print(device)
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# encoder: take a string, output a list of integers
def encode(s):
    return [stoi[c] for c in s]


# decoder: take a list of integers, output a string
def decode(l):
    return "".join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
    distmult=distmult,
)  # start with model_args from command line

conf = TuckerFormerConfig(**model_args)
model = TuckerFormer(conf)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
""" optimizer = Adafactor(
    model.parameters(),
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    lr=0.01,
    clip_threshold=1.0,
) """
for iter in tqdm(range(max_iters)):
    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        tqdm.write(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        # print(
        #    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        # )

        """ 
        print(f"The example is {decode(xb[0].tolist())}")
        print(yb[0])
        print("#" * 20)
        print(f"The prediction is {decode(torch.max(logits, dim=2)[1][0].tolist())}")
        print(torch.max(logits, dim=2)[1][0]) """


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
