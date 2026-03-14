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
            print(f"Помилка: знаменник tg(pi/2 - x) = tg({denom_arg:.4f}) не визначений (cos({denom_arg:.4f}) = 0).")
            return None
        tan_val = sin_val / cos_val
        if abs(tan_val) < 1e-10:
            print(f"Помилка: знаменник tg(pi/2 - x) = {tan_val:.6f} ≈ 0, ділення на нуль.")
            return None
        return numerator / tan_val

    elif 5 < x < 100:
        print("Використовується підфункція: y = sqrt(cos(x)) / ln(pow(x,2) - 36)")
        cos_val = math.cos(x)
        if cos_val < 0:
            print(f"Помилка: підкореневий вираз cos(x) = {cos_val:.4f} < 0, квадратний корінь не існує.")
            return None
        numerator = math.sqrt(cos_val)
        log_arg = x**2 - 36
        if log_arg <= 0:
            print(f"Помилка: аргумент логарифму (pow(x,2)-36) = {log_arg:.4f} <= 0, логарифм не визначений.")
            return None
        denominator = math.log(log_arg)
        if abs(denominator) < 1e-10:
            print(f"Помилка: знаменник ln(pow(x,2)-36) = {denominator:.6f} ≈ 0, ділення на нуль.")
            return None
        return numerator / denominator
    else:
        print(f"Помилка: значення x = {x} не входить до жодної з областей визначення функції.")
        print("Область визначення: x < 4  або  5 < x < 100")
        return None

def main():
    print("=" * 50)
    print("  Обчислення значення складної функції")
    print("=" * 50)
    print("  y = 5*sqrt(x+1) / tg(pi/2-x),  при x < 4")
    print("  y = sqrt(cos x) / ln(pow(x,2)-36),  при 5 < x < 100")
    print("=" * 50)

    try:
        x = float(input("Введіть значення аргументу x: "))
    except ValueError:
        print("Помилка: введено некоректне число.")
        return

    print()
    result = compute_function(x)

    if result is not None:
        print(f"\nРезультат: y({x}) = {result:.6f}")
    else:
        print("\nОбчислення неможливе.")

if __name__ == "__main__":
    main()
