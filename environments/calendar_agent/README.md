# calendar-agent

Calendar-agent is a Verifiers `StatefulToolEnv` for meeting scheduling. Each task asks an agent to choose a meeting day, UTC start time, room, and attendee set while balancing hard feasibility constraints and soft attendee utilities.

## Task Shape

The visible prompt gives the meeting duration, a calendar-day window within a 31-day month, global UTC working hours, required attendees, optional attendees, and rooms. Hidden task metadata stores the full generated calendar problem. Agents can reveal relevant pieces only through tools:

- `check_attendee_calendar(attendee)`
- `view_attendee_constraints(attendee="all")`
- `check_room_availability(room="all")`
- `check_score(day, start_time, room, attendees=None)`
- `submit_window(day, start_time, room, attendees=None)`

Tool responses include `remaining_turns`.

## Scoring

Submissions are acceptable only if they satisfy hard constraints:

- meeting is in the global window
- selected room exists and is free
- all required attendees are included
- every included required attendee is free and inside their hard local work window

Acceptable submissions receive a raw `calendar_reward`, computed as the weighted sum of attendee utilities. Attendee weights are generated per task and normalized to sum to `1.0`. Soft penalties include local time preference distance, disliked days, and back-to-back meeting penalties. Optional attendees can be omitted, but omission receives only a small utility.

The training reward is `normalized_score`: the submitted raw score divided by the best possible raw score for that generated task. Invalid or missing submissions still receive `0`. The generator enumerates all candidate days, 30-minute starts, rooms, and optional-attendee subsets to prove solvability and store the best possible score.

Attendee names and visible calendar-day windows are sampled deterministically from the task seed. Reusing the same seed recreates the same people and calendars, while adjacent generated tasks usually have different people, calendars, constraints, rooms, day ranges, and optima.

Room order is also shuffled deterministically per generated task so equivalent high-scoring choices do not always tie-break toward the same room name.

For final heldout testing, `split="heldout_generalization"` switches to a separate deterministic pool of attendee names, room names, and time zones. This changes the surface distribution without changing the scoring rules.

## Quickstart

```bash
prime env install calendar-agent
prime eval run calendar-agent -n 5 -r 1
```

With arguments:

```bash
prime eval run calendar-agent -n 10 -r 1 -a '{"difficulty":"hard","seed":13,"max_turns":16}'
```

Render a generated task in the terminal:

```bash
cd environments/calendar_agent
python calendar_tui.py --difficulty medium --seed 7
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `difficulty` | str | `"medium"` | One of `easy`, `medium`, or `hard`. Maps to attendee counts, window size, busy-block density, room count, and validity-ratio targets. |
| `num_examples` | int | `25` | Number of generated examples in the dataset. |
| `seed` | int | `7` | Base deterministic seed. |
| `max_turns` | int or null | preset-specific | Maximum model turns before the rollout ends. |
| `split` | str or null | null | Optional named split preset such as `train_easy`, `dev_easy`, `heldout_easy`, `train_medium`, `dev_medium`, `heldout_medium`, or `heldout_generalization`. Split presets override `difficulty`, `seed`, and generation profile. |
| `generation_profile` | str | `"standard"` | `standard` or `generalization`. Usually set through `split`. |
| `prompt_variant` | str | `"default"` | `default`, `brief`, `ticket`, `stakeholder`, or `mixed`. `mixed` deterministically rotates prompt templates across examples. |

## Metrics

| Metric | Meaning |
| --- | --- |
| `reward` | Training reward: normalized score, or `0` if no acceptable submission was made. |
| `calendar_reward` | Raw weighted attendee utility for the submitted meeting. |
| `normalized_score` | Submitted score divided by the best possible score for that task. |
| `optimum_gap` | Best possible score minus submitted score. |
| `found_acceptable` | `1` when the submitted window is hard-feasible, otherwise `0`. |
| `exact_optimal` | `1` when the submitted window reaches the oracle-normalized optimum. |
| `submitted_any` | `1` when the agent called `submit_window`. |
| `invalid_submission` | `1` when the agent submitted a hard-invalid window. |
| `task_attendee_count` | Average number of attendees in the evaluated tasks. |
| `task_optional_count` | Average number of optional attendees in the evaluated tasks. |
| `task_room_count` | Average number of rooms in the evaluated tasks. |
| `task_valid_ratio` | Average share of candidate choices that are hard-feasible. |
| `task_timezone_span_hours` | Average attendee timezone spread. |
| `slice_low_valid_density` | Share of tasks with low valid-solution density. |
| `slice_late_optimum` | Share of tasks whose oracle starts in the late part of the workday. |

The dataset also includes `task_slices`, `prompt_variant`, and `generation_profile` columns. Include them in `state_columns` for Prime eval runs when you want to inspect individual examples in the UI.
