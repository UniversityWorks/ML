import math

def compute_function(x):
    if x < 4:
        print("Використовується підфункція: y = 5·sqrt(x+1) / tg(pi/2 - x)")
        if x + 1 < 0:
            print(f"Помилка: підкореневий вираз (x+1) = {x+1:.4f} < 0, квадратний корінь не існує.")
            return None
        numerator = 5 * math.sqrt(x + 1)
        denom_arg = math.pi / 2 - x
        cos_val = math.cos(denom_arg)
        sin_val = math.sin(denom_arg)
        if abs(cos_val) < 1e-10:
            print(f"Помилка: знаменник tg(pi/2 - x) не визначений при x = {x}.")
            return None
        tan_val = sin_val / cos_val
        if abs(tan_val) < 1e-10:
            print(f"Помилка: знаменник tg(pi/2 - x) = {tan_val:.6f} ≈ 0.")
            return None
        return numerator / tan_val

    elif 5 < x < 100:
        print("Використовується підфункція: y = sqrt(cos x) / ln(pow(x,2) - 36)")
        cos_val = math.cos(x)
        if cos_val < 0:
            print(f"Помилка: підкореневий вираз cos(x) = {cos_val:.4f} < 0.")
            return None
        numerator = math.sqrt(cos_val)
        log_arg = x**2 - 36
        if log_arg <= 0:
            print(f"Помилка: аргумент логарифму (pow(x,2)-36) = {log_arg:.4f} ≤ 0.")
            return None
        denominator = math.log(log_arg)
        if abs(denominator) < 1e-10:
            print(f"Помилка: знаменник ln(pow(x,2)-36) ≈ 0.")
            return None
        return numerator / denominator
    else:
        print(f"Помилка: x = {x} не входить до жодної з областей визначення.")
        print("Область визначення: x < 4  або  5 < x < 100")
        return None

def main():
    print("=" * 55)
    print("  Обчислення значення складної функції (багатократне)")
    print("=" * 55)
    print("  y = 5 sqrt(x+1) / tg(pi/2-x),  при x < 4")
    print("  y = sqrt(cos x) / ln(pow(x,2)-36),  при 5 < x < 100")
    print("=" * 55)

    try:
        key_value = float(input("Введіть 'ключове' значення x для завершення програми: "))
    except ValueError:
        print("Помилка: некоректне ключове значення. Використовується 0.")
        key_value = 0.0

    print(f"Ключове значення: x = {key_value}")
    print("=" * 55)

    iteration = 1
    while True:
        print(f"\n--- Ітерація {iteration} ---")
        try:
            x_input = input("Введіть значення x (або ключове значення для виходу): ")
            x = float(x_input)
        except ValueError:
            print("Помилка: введено некоректне число. Спробуйте ще раз.")
            continue

        if x == key_value:
            print(f"\nВведено ключове значення x = {key_value}. Завершення програми.")
            break

        print()
        result = compute_function(x)

        if result is not None:
            print(f"Результат: y({x}) = {result:.6f}")
        else:
            print("Обчислення неможливе для даного x.")

        iteration += 1

    print("\nПрограму завершено.")

if __name__ == "__main__":
    main()
