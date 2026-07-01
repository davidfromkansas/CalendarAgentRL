# Calendar Agent Seed Splits

Use non-overlapping seed ranges so RL progress is measured on fresh calendar tasks.

| Split preset | Difficulty | Seed range | Profile | Purpose |
| --- | --- | --- | --- | --- |
| `train_easy` | easy | 10000+ | standard | RL updates for the cheap 4B budget recipe |
| `dev_easy` | easy | 20000+ | standard | Repeated progress checks while tuning |
| `heldout_easy` | easy | 30000+ | standard | Final check after choosing a recipe |
| `train_medium` | medium | 40000+ | standard | Harder RL updates after the easy recipe works |
| `dev_medium` | medium | 50000+ | standard | Medium progress checks while tuning |
| `heldout_medium` | medium | 60000+ | standard | Medium final check after choosing a recipe |
| `train_hard` | hard | 70000+ | standard | Advanced RL updates after medium is stable |
| `dev_hard` | hard | 80000+ | standard | Hard progress checks while tuning |
| `heldout_hard` | hard | 85000+ | standard | Hard final check after choosing a recipe |
| `heldout_generalization` | medium | 90000+ | generalization | Final generalization exam with heldout names, rooms, and time zones |

Rules:

- Do not train on dev or heldout seeds.
- Use dev eval for iteration and debugging.
- Use heldout eval sparingly after the recipe is fixed.
- Keep the same model, sampling settings, prompt-variant setting, and environment version when comparing runs.
- Use `prompt_variant = "mixed"` for rigorous evals so models do not overfit one public prompt template.
- Treat `heldout_generalization` as the blog-claim eval. Run it after the training recipe is fixed.
- Move from easy to medium to hard only after the dev split at the current difficulty is stable. Hard tasks have fewer score checks and fewer acceptable answers, so they are better as a later curriculum stage than a starting point.

This guards against models learning quirks of a repeated training set instead of learning the general tool workflow.
