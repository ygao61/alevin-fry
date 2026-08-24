#!/usr/bin/env python3
"""Generate resources/mlp_reference_kmers.tsv: PyTorch (f32) outputs of the Forseti
binding-affinity MLP on random 30-mers. Used by the NativeMlp unit test to prove the
hand-written evaluator reproduces the libtorch/PyTorch numbers.

Model: one-hot(30 x {A,C,G,T,N}) -> Linear(150,100) -> ReLU -> Linear(100,1) -> sigmoid,
weights from resources/mlp_params_Transpose.json (already stored as [out, in])."""
import json, random, torch, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
P = json.load(open(root / "resources/mlp_params_Transpose.json"))
W1 = torch.tensor(P["weights"][0], dtype=torch.float32)  # (100,150)
b1 = torch.tensor(P["biases"][0], dtype=torch.float32)
W2 = torch.tensor(P["weights"][1], dtype=torch.float32)  # (1,100)
b2 = torch.tensor(P["biases"][1], dtype=torch.float32)
code = {c: i for i, c in enumerate("ACGTN")}
def onehot(km):
    x = torch.zeros(150, dtype=torch.float32)
    for j, c in enumerate(km):
        x[j * 5 + code[c]] = 1.0
    return x
rng = random.Random(20260823)
kmers = ["A" * 30, "T" * 30, "N" * 30, "A" * 29 + "C"]
for _ in range(5000):
    alphabet = "ACGT" if rng.random() < 0.8 else "ACGTN"
    # bias toward A-rich k-mers, which is where the hot (6A) positions live
    p_a = rng.choice([0.25, 0.5, 0.8])
    kmers.append("".join("A" if rng.random() < p_a else rng.choice(alphabet) for _ in range(30)))
with torch.no_grad():
    X = torch.stack([onehot(k) for k in kmers])
    y = torch.sigmoid(torch.relu(X @ W1.T + b1) @ W2.T + b2).view(-1)
out = root / "resources/mlp_reference_kmers.tsv"
with open(out, "w") as f:
    for k, v in zip(kmers, y.tolist()):
        f.write(f"{k}\t{v:.9g}\n")
print("wrote", out, len(kmers))
