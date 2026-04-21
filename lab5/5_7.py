import pandas as pd

data = {
    "Товар": ["Навушники", "Зошит", "Чай", "Кросівки", "Яблуко", "Ноутбук"],
    "Ціна":  [149.90,       5.50,   12.75,   89.00,     29.99,   1299.00],
}

df = pd.DataFrame(data)

print("Вихідна таблиця:")
print(df)

df_sorted = df.sort_values(by="Ціна", ascending=True).reset_index(drop=True)

print("\nТаблиця, відсортована за ціною (зростання):")
print(df_sorted)
