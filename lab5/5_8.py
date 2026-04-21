import pandas as pd

K = 4           
TARGET_MONTH = "Липень"  
months    = ["Червень", "Червень", "Липень", "Липень", "Серпень", "Серпень"]
bodies    = ["Озеро",   "Річка",   "Озеро",  "Ставок", "Річка",   "Озеро" ]
catches   = [12,         8,         15,        6,        20,        10      ]

index = pd.MultiIndex.from_arrays([months, bodies], names=["Місяць", "Водойма"])
df = pd.DataFrame({"Кількість риби": catches}, index=index)

print("Журнал риболовлі")
print(df)

print(f"\nПерші {K} записів")
print(df.head(K))

print("\nЗагальна статистика вилову за все літо")
stats = df["Кількість риби"].agg(["mean", "min", "max"])
stats.index = ["Середнє", "Мінімум", "Максимум"]
print(stats)

print(f"\nСумарний вилов за {TARGET_MONTH}")
monthly_total = df.loc[TARGET_MONTH, "Кількість риби"].sum()
print(f"{TARGET_MONTH}: {monthly_total} риб")
