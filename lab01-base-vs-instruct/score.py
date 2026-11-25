import json
import sys
from pathlib import Path

from checker import extract_answer, is_correct

HERE = Path(__file__).parent


def score(prompts, samples, rankings):
    rows = []
    for q in prompts["exact"]:
        responses = samples[q["id"]]
        ranking = rankings.get(q["id"], [])
        if sorted(ranking) != list(range(1, len(responses) + 1)):
            raise ValueError(f"{q['id']}: rank each of responses 1-{len(responses)} exactly once, got {ranking}")
        correct = [i for i, r in enumerate(responses, 1) if is_correct(r, q["answer"])]
        rows.append({
            "id": q["id"],
            "expected": q["answer"],
            "extracted": [extract_answer(r) for r in responses],
            "correct": correct,
            "top": ranking[0],
            "verdict": "no correct response" if not correct
            else "all correct" if len(correct) == len(responses)
            else "agree" if ranking[0] in correct
            else "DISAGREE",
        })
    return rows


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python score.py results/<run>")
    run = Path(sys.argv[1])
    prompts = json.loads((HERE / "prompts.json").read_text())
    rows = score(prompts, json.loads((run / "samples.json").read_text()),
                 json.loads((run / "rankings.json").read_text()))
    lines = ["| Prompt | Expected | Extracted (1-4) | Correct | My top | Verdict |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['expected']} | {r['extracted']} | {r['correct']} | {r['top']} | {r['verdict']} |")
    report = "\n".join(lines) + "\n"
    (run / "score.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
