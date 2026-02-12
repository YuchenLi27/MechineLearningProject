import re

def simple_tokenize(text: str):
    # 中英混合时先用一个“够用”的切法：英文按词+符号，中文按单字
    # 你后面可以替换成更好的 tokenizer
    tokens = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":  # CJK
            tokens.append(ch)
        else:
            tokens.append(ch)
    text2 = "".join(tokens)

    # 英文/数字连续串、标点分离
    pattern = r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\s]"
    return re.findall(pattern, text2)

if __name__ == "__main__":
    s = "Today I was stressed 但是还好, let's go!"
    print(simple_tokenize(s))
