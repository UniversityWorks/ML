def task1():
    def count_digits(s):
        count = 0
        for ch in s:
            if ch.isdigit():
                count += 1
        return count

    s = input("Введіть рядок: ")
    print("Кількість цифр:", count_digits(s))


def task2():
    def count_substring(s, sub):
        count = 0
        i = 0
        while i <= len(s) - len(sub):
            if s[i:i+len(sub)] == sub:
                count += 1
            i += 1
        return count

    s = input("Введіть рядок: ")
    sub = input("Введіть підрядок: ")
    print("Кількість входжень:", count_substring(s, sub))


def task3():
    sentence = input("Введіть речення: ")
    words = sentence.split()
    unique = []
    for w in words:
        if w not in unique:
            unique.append(w)
    print("Унікальні слова:", unique)


def task4():
    def is_palindrome(word):
        word = word.lower()
        for i in range(len(word) // 2):
            if word[i] != word[len(word) - 1 - i]:
                return False
        return True

    word = input("Введіть слово: ")
    if is_palindrome(word):
        print("Паліндром")
    else:
        print("Не паліндром")


def task5():
    s = input("Введіть рядок: ")
    symbols = input("Введіть символи множини через пробіл: ").split()

    max_count = 0
    result = []

    for sym in symbols:
        count = 0
        for ch in s:
            if ch == sym:
                count += 1
        if count > max_count:
            max_count = count
            result = [sym]
        elif count == max_count and count > 0:
            result.append(sym)

    print("Найчастіші символи:", result)


def task6():
    n = int(input("Введіть число від 1 до 9: "))
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    for i in range(n):
        print(" ____  ", end="")
    print()

    for i in range(n):
        num = str(i + 1)
        letter = alphabet[i]
        print("| " + letter + "  " + num + " |", end="")
    print()

    for i in range(n):
        print("|      |", end="")
    print()

    for i in range(n):
        print("|      |", end="")
    print()

    for i in range(n):
        print("|___   |", end="")
    print()

    for i in range(n):
        print("|      |", end="")
    print()


def task7():
    def is_identifier(word):
        if len(word) == 0:
            return False
        first = word[0]
        if not (first.isalpha() or first == "_"):
            return False
        for ch in word[1:]:
            if not (ch.isalpha() or ch.isdigit() or ch == "_"):
                return False
        return True

    word = input("Введіть слово: ")
    if is_identifier(word):
        print("Ідентифікатор")
    else:
        print("Не ідентифікатор")


def task8():
    x = 0
    y = 0

    print("Вводьте рядки (завершіть рядком Treasure!):")
    while True:
        line = input()
        if line == "Treasure!":
            break
        parts = line.split()
        direction = parts[0]
        steps = int(parts[1])
        if direction == "North":
            y += steps
        elif direction == "South":
            y -= steps
        elif direction == "East":
            x += steps
        elif direction == "West":
            x -= steps

    print(x, y)


def task9():
    def mask_digits(s):
        if len(s) != 16:
            return "Рядок має бути 16 символів"
        last4 = s[12:]
        return "**** **** **** " + last4

    s = input("Введіть 16 цифр: ")
    print(mask_digits(s))


def task10():
    def reverse_words(sentence):
        words = sentence.split()
        new_words = []
        for w in words:
            new_words.append(w[::-1])
        return " ".join(new_words)

    sentence = input("Введіть речення: ")
    print(reverse_words(sentence))


def task11():
    text = input("Введіть текст: ")
    vowels = "аеєиіїоуюяaeiou"
    v_count = 0
    c_count = 0

    for ch in text.lower():
        if ch.isalpha():
            if ch in vowels:
                v_count += 1
            else:
                c_count += 1

    total = v_count + c_count
    if total > 0:
        print(f"Голосні: {v_count / total * 100:.1f}%")
        print(f"Приголосні: {c_count / total * 100:.1f}%")
    else:
        print("Немає літер")


def task12():
    def censor(text, banned):
        words = text.split()
        result = []
        for w in words:
            if w.lower() in [b.lower() for b in banned]:
                result.append("[ЦЕНЗУРА]")
            else:
                result.append(w)
        return " ".join(result)

    text = input("Введіть текст: ")
    banned = input("Введіть заборонені слова через пробіл: ").split()
    print(censor(text, banned))


def task13():
    def rle_encode(s):
        if len(s) == 0:
            return ""
        result = ""
        count = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                count += 1
            else:
                result += str(count) + s[i-1]
                count = 1
        result += str(count) + s[-1]
        return result

    def rle_decode(s):
        result = ""
        i = 0
        while i < len(s):
            num = ""
            while i < len(s) and s[i].isdigit():
                num += s[i]
                i += 1
            if i < len(s):
                result += s[i] * int(num)
                i += 1
        return result

    s = input("Введіть рядок: ")
    encoded = rle_encode(s)
    print("Закодований:", encoded)
    print("Декодований:", rle_decode(encoded))


def task14():
    row1 = input("Введіть перший рядок: ")
    row2 = input("Введіть другий рядок: ")

    counts = []
    positions = []

    for ch in row1:
        count = 0
        pos = []
        for i in range(len(row2)):
            if row2[i] == ch:
                count += 1
                pos.append(i)
        counts.append(count)
        positions.append(pos)

    print("Кількість входжень:", counts)
    print("Позиції:", positions)


tasks = {
    "1": ("Кількість цифр у рядку", task1),
    "2": ("Кількість входжень підрядка", task2),
    "3": ("Унікальні слова речення", task3),
    "4": ("Перевірка паліндрому", task4),
    "5": ("Символи множини що зустрічаються найчастіше", task5),
    "6": ("Літери алфавіту у вигляді стовпців", task6),
    "7": ("Перевірка ідентифікатора", task7),
    "8": ("Кіт Леопольд і скарб", task8),
    "9": ("Маскування 16 цифр", task9),
    "10": ("Речення з реверсованими словами", task10),
    "11": ("Відсоток голосних і приголосних", task11),
    "12": ("Цензура заборонених слів", task12),
    "13": ("Run-Length Encoding і декодування", task13),
    "14": ("Входження символів першого рядка у другий", task14),
}

print()

while True:
    print("Оберіть завдання:")
    for key, (name, _) in tasks.items():
        print(f"  {key}. {name}")
    print("  0. Вихід")
    print()

    choice = input("Ваш вибір: ").strip()

    if choice == "0":
        break
    elif choice in tasks:
        print()
        print(f"Завдання {choice}: {tasks[choice][0]} ")

        tasks[choice][1]()
        print()
    else:
        print("Невірний вибір, спробуйте ще раз.")
        print()
