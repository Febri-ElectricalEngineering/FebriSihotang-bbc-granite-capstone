from textwrap import dedent

def classify_prompt(text: str, label_set=("business","entertainment","politics","sport","tech")) -> str:
    labels = ", ".join(label_set)
    return dedent(f"""\
        You are a precise news topic classifier. Read the article and choose ONE label only.
        Allowed labels: {labels}

        Article:
        \"\"\"{text.strip()}\"\"\"


        Answer with only the label name (no extra words).
    """)

def summarize_prompt(text: str, max_words: int = 60) -> str:
    return dedent(f"""\
        Summarize the article in at most {max_words} words.
        Keep neutral, factual tone. Avoid opinions and numbers not in the text.

        Article:
        \"\"\"{text.strip()}\"\"\"


        Summary:
    """)
