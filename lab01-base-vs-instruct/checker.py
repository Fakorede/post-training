import re

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text):
    """Return the number after the last 'answer is/:', else the last number, else None."""
    marked = re.findall(r"answer(?:\s+is|\s*:)\s*\$?(" + NUMBER.pattern + ")", text, re.I)
    found = marked or NUMBER.findall(text)
    return found[-1].replace(",", "").rstrip(".") if found else None


def is_correct(text, expected):
    got = extract_answer(text)
    return got is not None and float(got) == float(expected)
