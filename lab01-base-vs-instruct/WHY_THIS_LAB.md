# Lab 1: what's the point of this exercise?

It lets you see the pieces of the Week 1 pipeline diagram in action.

| What you did | What it shows | Why it matters later |
|---|---|---|
| Ran a base and an instruct model | How post-training changes behavior (see `comparison.md`) | It is what SFT and later stages aim to change. |
| Took 4 samples per prompt | One model can give both right and wrong answers. | RL depends on this: it makes better-scoring samples more likely. |
| Ranked the responses | You created preference judgments. | This is the kind of data reward models and DPO learn from. |
| Ran the checker | A program scored answers without your opinion. | This is the idea behind RLVR rewards. |

## The most useful result: e4

In run 1203372, the checker marked all four responses to "What is 100 minus 37?" wrong.

- **Preference ranking** still picks a "best" one, even if every answer is wrong.
- **A checker** gives all four the same zero. GRPO compares samples within a group, so identical
  rewards give no signal about which one to favor. This returns in Weeks 10-12.

## What it did not show

A clear case where preference and correctness disagreed. The checker marked the top choice correct
on all four questions that had both a chosen response and a correct one (e1, e2, e3, e5). This
arithmetic is easy to check by hand, so agreement is not surprising. Seeing a disagreement would
likely take harder questions, where an answer can sound convincing but be wrong.

## Practical check

The environment, a CPU job session and loading both models work on LONI. The GPU and training were
not tested by this run. Lab 2 is the first training run.

## Takeaway

A preference tells you which answer someone likes more. A checker tests an answer against a rule.
Neither is guaranteed to be right: this checker only picks a number out of the response.
