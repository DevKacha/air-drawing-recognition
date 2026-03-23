import pandas as pd
import numpy as np

try:
    df_shapes = pd.read_excel('d:/DL_Project/Shapes/Excel/label.xlsx', header=None)
    print("Shapes Labels (Unique):", df_shapes[0].unique(), df_shapes[0].nunique())
except Exception as e:
    print("Error reading Shapes:", e)

try:
    df_numbers = pd.read_excel('d:/DL_Project/Numbers/Excel/label.xlsx', header=None)
    print("Numbers Labels (Unique):", df_numbers[0].unique(), df_numbers[0].nunique())
except Exception as e:
    print("Error reading Numbers:", e)
