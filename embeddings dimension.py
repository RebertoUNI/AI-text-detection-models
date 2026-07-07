import numpy as np
data = np.load("test__emb__chunk00000.npy")
print(data.shape)  # → (100000, DIM) — DIM è il tuo input_dim