import pandas as pd

K = 5

data = {
    "Нейронні мережі": [90, 85, 78, 92, 88][:K],
    "СШІ":             [75, 80, 95, 70, 85][:K],
}

df = pd.DataFrame(data, index=[f"Студент {i+1}" for i in range(K)])

print(f"DataFrame з оцінками {K} студентів:")
print(df)
