# Calendar Agent Seed Splits

Use non-overlapping seed ranges so RL progress is measured on fresh calendar tasks.

| Split | Difficulty | Seed range | Purpose |
| --- | --- | --- | --- |
| Train | easy | 10000-10127 | RL updates for the cheap 4B budget recipe |
| Dev eval | easy | 20000-20099 | Repeated progress checks while tuning |
| Heldout eval | easy | 30000-30099 | Final check after choosing a recipe |

Rules:

- Do not train on dev or heldout seeds.
- Use dev eval for iteration and debugging.
- Use heldout eval sparingly after the recipe is fixed.
- Keep the same model, sampling settings, and environment version when comparing runs.

This guards against models learning quirks of a repeated training set instead of learning the general tool workflow.
