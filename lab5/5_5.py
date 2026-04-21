import pandas as pd

K = 3
L = 2

data = {
    "Товар": ["Яблуко", "Банан", "Молоко", "Хліб", "Сир", "Масло", "Сік"],
    "Ціна":  [12.5,     8.0,    35.9,    18.0,  65.0,  55.5,  42.0],
    "К-сть": [100,      80,      60,      200,    40,    50,    90],
}

df = pd.DataFrame(data)

print("Повний DataFrame:")
print(df)

print(f"\nПерші {K} рядки (head):")
print(df.head(K))

print(f"\nОстанні {L} рядки (tail):")
print(df.tail(L))
