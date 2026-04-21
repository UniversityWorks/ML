import pandas as pd
import os

csv_path = "education.csv"

if not os.path.exists(csv_path):
    sample = pd.DataFrame({
        "Name":    ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "Subject": ["Math", "Physics", "Math", "Chemistry", "Physics"],
        "Grade":   [88, 74, 95, 61, 83],
    })
    sample.to_csv(csv_path, index=False)
    print(f"(Файл '{csv_path}' не знайдено — створено зразковий файл.)\n")

df = pd.read_csv(csv_path)

print(f"Файл: {csv_path}")
print(f"Кількість рядків:   {df.shape[0]}")
print(f"Кількість стовпців: {df.shape[1]}")
print(f"\n(Властивість shape: {df.shape})")
