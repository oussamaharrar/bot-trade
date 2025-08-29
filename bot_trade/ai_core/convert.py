def azerty_to_arabic(text):
    mapping = {
        'a': 'ش', 'z': 'ص', 'e': 'ث', 'r': 'ق', 't': 'ف', 'y': 'غ', 'u': 'ع', 'i': 'ه', 'o': 'خ', 'p': 'ح',
        'q': 'س', 's': 'د', 'd': 'ج', 'f': 'ل', 'g': 'ك', 'h': 'م', 'j': 'ن', 'k': 'ت', 'l': 'ن', 'm': 'م',
        'ù': 'ك', '*': 'ط',
        'w': 'ز', 'x': 'ظ', 'c': 'ط', 'v': 'ذ', 'b': 'ئ', 'n': 'ء', ',': 'ؤ', ';': 'ر', ':': 'لا', '!': 'ى'
    }

    converted = ''.join(mapping.get(char, char) for char in text)
    return converted


if __name__ == "__main__":
    print("🔤 أدخل النص الذي كتبته بلوحة مفاتيح AZERTY (فرنسية):")
    input_text = input(">> ")
    output_text = azerty_to_arabic(input_text)
    print("\n✅ النص بعد التحويل للعربية:")
    print(output_text)
