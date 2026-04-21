import pandas as pd

data = {
    "Нейронні мережі": [90, 85, 78, 92, 88, 70],
    "СШІ":             [75, 80, 95, 70, 85, 60],
}

students = [f"Студент {i+1}" for i in range(6)]
df = pd.DataFrame(data, index=students)

print("Таблиця оцінок студентів:")
print(df)

print("\nСередній бал для кожної дисципліни:")
print(df.mean().round(2))
