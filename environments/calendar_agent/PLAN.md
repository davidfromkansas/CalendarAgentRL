# PLAN

## Design

Build a Prime/Verifiers environment where the agent schedules a meeting from realistic partial information. The environment is intentionally deterministic: task generation creates a hidden full problem, enumerates all candidate choices, validates that a satisfying solution exists, and records the optimum score before the task reaches the model.

The implementation uses `vf.StatefulToolEnv` because each rollout needs task-local state: hidden calendar data, submitted choice, score results, and remaining-turn feedback. All data is in memory and JSON-serializable.

## Composition Rules

Hard constraints zero the score if violated:

- global day and UTC work window
- room existence and availability
- inclusion of required attendees
- required attendee calendar conflicts
- required attendee hard local time windows

Soft constraints reduce individual utility:

- distance from local preferred time window
- disliked days
- back-to-back adjacency to existing meetings
- optional attendee omission

Raw calendar utility is the weighted average of attendee utilities. Attendee weights are normalized to `1.0` per task. The RL reward is the normalized score: submitted raw utility divided by the task's best possible raw utility, with invalid or missing submissions receiving `0`.

## Backdoor Resistance

- Hidden task metadata is never shown in the prompt.
- The best choice and optimum score are stored only in `info`, not visible to the agent.
- Tools expose realistic views: attendee calendars, attendee constraints, room availability, candidate score, and final submission.
- Scoring recomputes deterministically from the hidden problem spec rather than trusting tool text or model output.
- `max_turns` limits exhaustive probing, while `check_score` still allows honest exploration.

## Generation Strategy

Difficulty presets map high-level labels to fine-grained ranges:

- attendee count
- number of days
- busy-block count
- optional attendee count
- room count
- target valid-choice ratio
- minimum optimum score
- default max turns

Generation samples candidate problems and filters by exact enumeration. This keeps most shipped tasks solvable, avoids trivial all-open calendars, and makes random start-time proposals a poor strategy.

## Testing Plan

1. Import the package and call `load_environment()`.
2. Generate small datasets for easy, medium, and hard.
3. Assert attendee weights sum to `1.0`.
4. Assert each generated problem has a best choice and positive best score.
5. Assert the recorded best choice rescoring matches the recorded optimum.
6. Run the Rich TUI script for one sample.
7. Install with `prime env install calendar-agent`.
8. Run a small `prime eval run calendar-agent` smoke test with the configured model.

## Milestone Status

- Completed: Prime scaffold created with `prime env init calendar-agent`.
- Completed: StatefulToolEnv implementation with deterministic generator, solver, tools, and normalized RL reward.
- Completed: Standalone Rich TUI script.
- Completed: README and initial PLAN.
- Completed: Import/runtime checks for `load_environment()`.
- Completed: Easy, medium, and hard generation checks for positive optimums, valid ratios, normalized weights, and exact best-choice rescoring.
- Completed: Tool schema check confirmed hidden rollout `state` is not exposed to the agent.
- Completed: Rich TUI smoke render for an easy sample.
- Completed: Installed with `prime env install calendar-agent`.
- Completed: Added and ran `test_calendar_agent.py` directly with the workspace Python; all tests passed.
- Completed: Ran two one-example Prime eval smoke tests with `Qwen/Qwen3.5-4B`; both loaded, served, saved results, and shut down cleanly.
- Observed: The small Qwen baseline over-analyzed and did not call `submit_window` before truncation, yielding reward `0`. This is a model/prompt-ergonomics baseline issue rather than an environment loading or scoring failure. Future tuning should try stronger instruction models, lower reasoning verbosity, or a concise baseline policy prompt.
- Completed: Clarified UTC handling after GPT-4.1-mini showed hard-conflict failures from local/UTC confusion. Attendee calendar tools now expose `busy_blocks_utc`, and the system prompt requires `check_score(acceptable=true)` before `submit_window`.
- Completed: GPT-4.1-mini easy eval after the clarity patch produced successful non-zero rollouts on 2/3 examples, with `found_acceptable=0.667` and average reward `0.441`.
- Completed: GPT-4.1 easy eval produced 3/3 acceptable submissions, average reward `0.748`, and average normalized score `0.901`.
- Completed: GPT-4.1 medium eval produced 2/3 acceptable submissions, average reward `0.470`, and one max-turns failure after repeated score probes. This gives useful reward diversity for RL while confirming the environment is solvable by a stronger model.
- Completed: Switched the RL reward from raw `calendar_reward` to `normalized_score`, while preserving raw utility, optimum gap, and acceptability as metrics.
- Completed: Expanded and seed-shuffled attendee names so generated tasks do not all appear to involve the same people while preserving deterministic reproducibility.
- Completed: Seed-shuffled room order per task to reduce fixed `Atlas` tie-breaking in optimal choices.
- Completed: Created Hosted Training config `configs/rl/calendar-agent-qwen-35b.toml` for `Qwen/Qwen3.5-35B-A3B`, `batch_size=128`, and `rollouts_per_example=8`.
- Completed: Started Hosted Training run `dw7z1ynkrl3jhz4ruifykcrr`; it later failed during startup because the config used duplicate evaluation environment names.
- Completed: Patched `configs/rl/calendar-agent-qwen-35b.toml` to give the train/eval environments unique names.
- Completed: Started replacement Hosted Training run `hke23mlkkdc7yls5cr9mvex1`; status and components were `RUNNING` after startup.
- Observed: Replacement run `hke23mlkkdc7yls5cr9mvex1` failed while the orchestrator loaded training environments. Raw logs show `ModuleNotFoundError: No module named 'calendar_agent'`, followed by `Could not import 'calendar-agent' environment. Ensure the package for the 'calendar-agent' environment is installed.`
- Blocked: Hosted Training needs the environment package available to the orchestrator, which requires publishing the environment to Prime Hub or another supported install path. Publishing is currently blocked because the configured team has no slug/teamname, and the personal account has no public username.
- Completed: Published the environment privately as `davidfromkansas/calendar-agent`.
- Completed: Updated `configs/rl/calendar-agent-qwen-35b.toml` to use the published Hub ID.
- Completed: Started Hosted Training run `af9nqyslfd2mqq9pdk883nrf`; logs confirmed the orchestrator pulled and installed `davidfromkansas/calendar-agent@0.1.0`, and status was `RUNNING`.
- Observed: Run `af9nqyslfd2mqq9pdk883nrf` was automatically stopped when wallet balance was exhausted.
- Completed: After wallet balance was restored, launched fresh Hosted Training run `smtsimdsia3lgfuplt2sskzc`; status was `RUNNING` and logs confirmed the orchestrator installed `davidfromkansas/calendar-agent@0.1.0`.
