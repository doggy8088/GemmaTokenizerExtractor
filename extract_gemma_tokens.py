from transformers import GemmaTokenizer

def extract_gemma_tokens(model_name="./gemma-2-9b-it"):
    tokenizer = GemmaTokenizer.from_pretrained(model_name)
    all_tokens = list(tokenizer.get_vocab().keys())
    print(f"Gemma 模型中共有 {len(all_tokens)} 個 token")

    # 寫入 tokens 到檔案
    with open("tokens.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(all_tokens))

if __name__ == "__main__":
    extract_gemma_tokens()