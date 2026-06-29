# CalendarAgentRL

CalendarAgentRL is a Prime/Verifiers reinforcement learning environment for training and evaluating calendar scheduling agents. It creates synthetic meeting-scheduling problems with hidden calendars, attendee preferences, room availability, hard constraints, soft utility tradeoffs, and deterministic oracle scoring.

The goal is to train models to behave like practical scheduling agents: inspect the right information with tools, compare candidate meeting windows, and submit a high-quality feasible meeting time before the turn limit.

## What The Environment Tests

Each task asks an agent to schedule one meeting. The visible prompt includes the public setup:

- meeting duration
- allowed calendar-day range inside a month
- allowed UTC working hours
- required attendees
- optional attendees
- available rooms

The actual calendar data is hidden behind tools. The model must discover the relevant details through tool calls instead of seeing the full problem upfront.

This makes the environment test agent behavior rather than simple text completion. A good rollout usually looks like:

```text
inspect attendee constraints
inspect calendars
inspect room availability
check a few candidate windows
submit the best acceptable window
```

## Tools Available To The Agent

The environment is implemented as a Verifiers `StatefulToolEnv`. The agent receives these tools:

| Tool | Purpose |
| --- | --- |
| `view_attendee_constraints(attendee="all")` | Reveals hard and soft constraints for one attendee or all attendees. |
| `check_attendee_calendar(attendee)` | Shows one attendee's busy calendar blocks in UTC and local time. |
| `check_room_availability(room="all")` | Shows room busy blocks in UTC. |
| `check_score(day, start_time, room, attendees)` | Scores a candidate meeting without submitting it. |
| `submit_window(day, start_time, room, attendees)` | Makes the final answer for the rollout. |

Tool responses include remaining turns, so the model has to manage its exploration budget.

## Hard And Soft Constraints

Hard constraints make a submission invalid and give reward `0`. Examples:

- required attendee omitted
- required attendee has a calendar conflict
- meeting is outside global UTC working hours
- meeting is outside a required attendee's hard local work window
- selected room does not exist or is unavailable
- invalid day or non-30-minute start time

Soft constraints affect utility without making the meeting invalid. Examples:

- attendee local-time preference distance
- disliked days
- back-to-back meeting penalties
- optional attendee omission

Each attendee has an importance weight. Weights are normalized to sum to `1.0` per task.

## Reward Design

The environment computes two important scores:

| Score | Meaning |
| --- | --- |
| `calendar_reward` | Raw weighted attendee utility for the submitted meeting. |
| `normalized_score` | Submitted raw utility divided by the best possible raw utility for that generated task. |

The RL training reward is `normalized_score`.

That means an optimal solution receives `1.0` even if the best real-world compromise has raw utility below `1.0`. Invalid or missing submissions still receive `0`.

This is useful because some generated scheduling problems are inherently constrained. A meeting with raw utility `0.76` may be the best possible answer for that specific task, and the model should receive full credit for finding it.

The environment still records raw utility as `calendar_reward` so you can inspect absolute meeting quality.

## Deterministic Generation And Oracle Validation

Tasks are generated programmatically from a seed. Each task samples a visible calendar-day window within a 31-day month, such as days `8-10` or `24-27`, instead of always using relative days `0-2`. For each generated task, the environment enumerates all candidate choices:

- days
- 30-minute start times
- rooms
- optional-attendee subsets

It then computes the best possible feasible meeting and stores that oracle score in hidden task metadata. The model cannot see the oracle answer, but the evaluator uses it to compute `normalized_score`, `optimum_gap`, and validation metrics.

Generation is deterministic:

```text
same difficulty + same seed -> same task
different seed -> different task
```

Attendee names and room order are sampled deterministically per task. This prevents every example from looking like the same group of people and avoids tie-breaking all optimal room choices toward the first room name.

For final generalization testing, the environment also supports a `generalization` generation profile with a heldout pool of attendee names, room names, and time zones. This keeps the scheduling rules the same while changing the surface distribution the model sees.

## Difficulty Levels

The environment supports:

- `easy`
- `medium`
- `hard`

Difficulty controls ranges for:

- attendee count
- optional attendee count
- number of scheduling days
- busy-calendar density
- room count
- target valid-solution ratio
- minimum accepted oracle score
- default max turns

## Splits, Prompt Variants, And Slice Metrics

`load_environment()` supports named split presets so training, dev, and final eval tasks can be kept separate:

| Split preset | Purpose |
| --- | --- |
| `train_easy` / `train_medium` | Training-only seeds. |
| `dev_easy` / `dev_medium` | Repeated debugging and recipe-selection evals. |
| `heldout_easy` / `heldout_medium` | Final standard-distribution evals. |
| `heldout_generalization` | Final generalization eval with heldout names, rooms, and time zones. |

For robust evals, use `prompt_variant="mixed"`. The environment renders equivalent public task prompts in several deterministic styles, including a compact brief, a ticket format, and a stakeholder-style request. The hidden calendars and scoring are still generated entirely by code.

Each task also exposes slice metadata in the dataset column `task_slices`, and the rubric reports task/context metrics such as:

- `task_attendee_count`
- `task_optional_count`
- `task_room_count`
- `task_valid_ratio`
- `task_timezone_span_hours`
- `slice_low_valid_density`
- `slice_late_optimum`
- `submitted_any`
- `invalid_submission`
- `exact_optimal`

These make it possible to inspect where a trained model improves or lags instead of relying only on average reward.

## Repository Layout

```text
environments/calendar_agent/
  calendar_agent.py          # StatefulToolEnv, generator, tools, scoring
  calendar_tui.py            # Rich terminal visualizer for generated tasks
  test_calendar_agent.py     # Local environment tests
  PLAN.md                    # Design notes, milestones, TODOs
  README.md                  # Environment-specific usage notes

configs/eval/
  calendar-agent-qwen-4b-dev-100.toml
  calendar-agent-gpt55-reference-20.toml
  calendar-agent-heldout-100.toml

configs/rl/
  calendar-agent-qwen-4b-budget.toml
  calendar-agent-qwen-35b.toml
  calendar-agent-splits.md
```

## Quickstart

Install the environment from the local project:

```bash
prime env install ./environments/calendar_agent
```

Run a small eval:

```bash
prime eval run davidfromkansas/calendar-agent \
  -m Qwen/Qwen3.5-4B \
  -n 10 \
  -r 1 \
  -t 2048 \
  -T 0.2 \
  -a '{"difficulty":"easy","num_examples":10,"seed":401,"max_turns":10}' \
  --disable-tui \
  --save-results \
  -C submitted_choice,submitted_result
```

Run the local tests:

```bash
uv run --project environments/calendar_agent python environments/calendar_agent/test_calendar_agent.py
```

View a generated problem in the terminal:

```bash
uv run --project environments/calendar_agent python environments/calendar_agent/calendar_tui.py --difficulty medium --seed 5
```

## Fixed Splits For RL

The project defines non-overlapping seed ranges so training progress is measured on fresh tasks:

| Split preset | Difficulty | Seed range | Purpose |
| --- | --- | --- | --- |
| `train_easy` | easy | `10000+` | RL updates for the cheap 4B budget recipe |
| `dev_easy` | easy | `20000+` | Repeated progress checks while tuning |
| `heldout_easy` | easy | `30000+` | Final check after choosing a recipe |
| `heldout_generalization` | medium | `90000+` | Final generalization check after choosing a recipe |

See [configs/rl/calendar-agent-splits.md](configs/rl/calendar-agent-splits.md).

## Current Baseline Findings

Early evals showed a strong separation between models:

- `openai/gpt-5.5` solved 10/10 easy tasks acceptably and reached average normalized reward around `0.994`.
- `Qwen/Qwen3.5-4B` showed basic tool use but solved only 1/10 in one easy eval.
- `Qwen/Qwen3.5-9B` used tools but got stuck analyzing and did not submit in the tested 10-task eval.

The main Qwen failure mode was not calendar reasoning alone. It was agent control: gather information, sometimes score candidates, but fail to call `submit_window` before stopping or hitting the turn limit.

This makes the environment a useful target for improving tool-use discipline with SFT, RL, or SFT followed by RL.

## Published Environment

The environment has been pushed privately on Prime Hub as:

```text
davidfromkansas/calendar-agent
```

Install with:

```bash
prime env install davidfromkansas/calendar-agent
```

## Notes

This is a research/prototyping environment, not a production calendar scheduler. The calendars are synthetic, generated from seeds, and contain no real user calendar data.
