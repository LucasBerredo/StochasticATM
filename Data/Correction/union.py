import pandas as pd

# Cargar ambos
df1 = pd.read_csv('Data/dataset_reparado.csv')
df2 = pd.read_csv('Data/dataset_reparado_v2.csv')

# Concatenar (uno debajo del otro)
df_final = pd.concat([df1, df2], ignore_index=True)

# Guardar
df_final.to_csv('Data/Datasets/dataset_total_unido.csv', index=False)