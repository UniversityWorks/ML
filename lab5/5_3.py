import pandas as pd

prices = pd.Series(
    [29.99, 149.90, 5.50, 89.00, 12.75],
    index=["Яблуко", "Навушники", "Зошит", "Кросівки", "Чай"],
)

print("Series з цінами на товари:")
print(prices)

label = "Навушники"
print(f"\nЗначення за текстовою міткою '{label}': {prices[label]}")

pos = 3
print(f"Значення за числовим індексом [{pos}]: {prices.iloc[pos]}")
