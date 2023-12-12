import pandas as pd
import numpy as np

# 4-boyutlu NumPy dizisi oluştur
z = np.arange(81).reshape(3, 3, 3, 3)

# Her bir üçüncü boyutu ayrı bir DataFrame'e dönüştür
dfs = [pd.DataFrame(z[:, :, i, :].reshape(-1, z.shape[-1])) for i in range(z.shape[2])]

# DataFrameleri birleştir
result_df = pd.concat(dfs, axis=1)

# CSV dosyasına yaz
result_df.to_csv('z.csv', index=False)

indices = (1,1,1,1)
z[indices]