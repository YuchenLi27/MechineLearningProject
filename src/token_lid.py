import fasttext
from src.tokenize_simple import simple_tokenize

MODEL_PATH = "models/lid.176.bin"

def lid_token(model, tok: str) -> str:
    # fastText 对单个字符/短 token 容易不稳：做几个规则兜底
    if tok.strip() == "":
        return "other"
    # 中文字符直接判 zh
    if "\u4e00" <= tok <= "\u9fff":
        return "zh"
    # 纯标点/符号
    if all(not c.isalnum() for c in tok):
        return "other"

    label, prob = model.predict(tok)
    lang = label[0].replace("__label__", "")
    # 只关心你们 pair（比如 en/zh）；其它都归 other
    if lang.startswith("en"):
        return "en"
    if lang.startswith("zh"):
        return "zh"
    return "other"

if __name__ == "__main__":
    model = fasttext.load_model(MODEL_PATH)
    text = "Today I was stressed 但是还好, let's go!"
    toks = simple_tokenize(text)
    langs = [lid_token(model, t) for t in toks]
    for t, l in zip(toks, langs):
        print(f"{t}\t{l}")
