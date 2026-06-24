# Calendar Agent Pseudocode

This document explains `calendar_agent.py` in plain pseudocode. It is meant for a semi-technical reader who wants to understand what the RL environment does without reading every Python detail.

## Big Picture

The environment creates synthetic calendar scheduling tasks.

For each task:

1. Generate fake people, calendars, rooms, and preferences.
2. Compute the best possible meeting by brute force.
3. Show the model only a short public prompt.
4. Let the model use tools to inspect hidden calendar data.
5. Grade the model's final submitted meeting deterministically.

The model is rewarded for finding the best feasible meeting for that generated task.

## Constants And Presets

The file starts by defining reusable ingredients:

```text
slot size = 30 minutes
workday starts at 09:00 UTC
possible room names = Atlas, Borealis, Cascade
possible attendee names = Avery, Blair, Casey, ...
possible time zones = Los Angeles, Denver, Chicago, New York, London
```

It also defines difficulty presets:

```text
easy:
  fewer attendees
  fewer busy calendar blocks
  more valid solutions

medium:
  more attendees
  denser calendars
  fewer valid solutions

hard:
  most attendees
  tightest calendars
  fewest valid solutions
```

Each preset controls ranges for:

```text
number of attendees
number of days
number of busy blocks
number of optional attendees
number of rooms
acceptable valid-solution ratio
minimum best possible score
default max turns
```

## Time Helpers

The code stores times as minutes after midnight.

Example:

```text
09:00 -> 540
13:30 -> 810
```

Pseudocode:

```text
minutes_to_time(minutes):
  convert 810 into "13:30"

time_to_minutes("13:30"):
  convert "13:30" into 810
  reject times that are not on 30-minute boundaries

overlaps(start, end, busy_block):
  return true if the meeting overlaps that busy block

local_minutes(utc_minute, timezone_offset):
  convert UTC time into local clock time

format_block(block, timezone_offset):
  format a busy block as day/start/end strings
```

## Attendee Utility

`attendee_utility(...)` computes how happy one attendee is with a proposed meeting.

Pseudocode:

```text
attendee_utility(attendee, day, start, end, included):
  if attendee is not included:
    if attendee is required:
      return 0
    else:
      return small optional-omission utility

  if meeting overlaps attendee busy calendar:
    return 0

  convert meeting start/end into attendee local time

  if meeting is outside attendee hard local work window:
    return 0

  start with utility = 1.0

  subtract preference cost:
    bigger penalty if meeting is far from preferred local time window

  subtract back-to-back cost:
    penalty if meeting is adjacent to another busy block

  subtract bad-day cost:
    penalty if attendee dislikes this day

  clamp final utility to at least 0
  return utility
```

Important idea:

```text
Hard conflicts return 0.
Soft preferences reduce utility but do not automatically invalidate the meeting.
```

## The Main Verifier: `score_choice`

`score_choice(problem, choice)` is the core verifier. It checks whether a proposed meeting is valid and computes its raw score.

Input:

```text
problem:
  generated calendar task

choice:
  day
  start_time
  room
  attendees
```

Pseudocode:

```text
score_choice(problem, choice):
  parse day, start time, duration, end time, room, included attendees
  violations = empty list

  if day is outside scheduling window:
    add violation

  if meeting is outside global workday:
    add violation

  if room does not exist:
    add violation
  else if room is busy during meeting:
    add violation

  attendee_scores = []

  for each attendee:
    is_included = attendee name is in included attendees
    utility = attendee_utility(attendee, day, start, end, is_included)

    if attendee is required and not included:
      add violation

    if attendee is required and included and utility is 0:
      add violation

    attendee_scores.append:
      name
      included?
      required?
      importance weight
      utility
      weight * utility

  acceptable = no violations

  if acceptable:
    score = sum(weighted utilities)
  else:
    score = 0

  normalized_against_optimum = score / problem.best_score

  return:
    acceptable
    score
    violations
    attendee_scores
    normalized_against_optimum
```

This is the main grading logic. It is deterministic code, not an LLM judge.

## Enumerating Candidate Meetings

`enumerate_choices(problem)` creates every meeting the model could theoretically submit.

Pseudocode:

```text
enumerate_choices(problem):
  required = all required attendee names
  optional = all optional attendee names

  attendee_subsets =:
    required attendees only
    required + each possible optional subset

  choices = []

  for each day:
    for each 30-minute start time in the global workday:
      for each room:
        for each attendee subset:
          add choice(day, start_time, room, attendees)

  return choices
```

This lets the environment prove what the best possible answer is.

## Solving The Generated Problem

`solve_problem(problem)` finds the oracle best meeting.

Pseudocode:

```text
solve_problem(problem):
  all_choices = enumerate_choices(problem)
  best_choice = none
  best_score = none
  valid_count = 0

  for each choice in all_choices:
    result = score_choice(problem, choice)

    if result is acceptable:
      valid_count += 1

    if this result has the highest score so far:
      best_choice = choice
      best_score = result.score

  return:
    best_choice
    best_score
    valid_count
    total candidate count
    valid_ratio = valid_count / total candidate count
```

This is how the environment knows whether the model found an optimal answer.

## Generating A Synthetic Calendar Problem

`build_problem(seed, difficulty)` creates one task.

Pseudocode:

```text
build_problem(seed, difficulty):
  preset = difficulty settings

  try up to 600 generation attempts:
    create deterministic random generator from seed + attempt

    sample:
      attendee count
      optional attendee count
      number of days
      meeting duration
      room count
      room names in shuffled order

    sample attendee weights
    normalize weights to sum to 1

    sample attendee names in shuffled order

    for each attendee:
      sample city and time zone
      sample hard local work window
      sample preferred local time window
      sample busy calendar blocks
      sample back-to-back penalty
      sample bad days and bad-day penalty
      sample optional omission utility

    for each room:
      sample capacity
      sample busy room blocks

    assemble problem

    solution = solve_problem(problem)

    if problem is good enough:
      best_choice exists
      best_score is above difficulty minimum
      valid_ratio is inside target range
      attach solution to problem
      return problem

  if no attempt satisfies all filters:
    return the last generated problem with its solution
```

The filters avoid tasks that are impossible, too easy, or too low-quality.

## Public Prompt Construction

`public_summary(problem)` creates the prompt shown to the model.

The prompt includes:

```text
meeting duration
day range
global UTC time window
required attendees
optional attendees
available rooms
instruction to inspect tools and submit one meeting
```

The prompt does not reveal:

```text
busy calendars
local time zones
soft preferences
room busy blocks
best answer
best possible score
```

Those details are hidden behind tools.

## Building A Dataset

`build_dataset(num_examples, difficulty, seed, max_turns)` creates many tasks.

Pseudocode:

```text
build_dataset(num_examples, difficulty, seed, max_turns):
  rows = []

  for index from 0 to num_examples - 1:
    problem = build_problem(seed + index, difficulty)
    problem.max_turns = max_turns

    row =:
      example_id = index
      question = public_summary(problem)
      answer = JSON with oracle best choice and score
      info.problem = full hidden problem

    rows.append(row)

  return Dataset(rows)
```

Important:

```text
The model sees question.
The environment keeps info.problem hidden for tools and scoring.
```

## Reward Metrics

The environment records several scoring functions.

```text
calendar_reward(state):
  if no submitted result:
    return 0
  return raw submitted score

optimum_gap(state):
  if no submitted result:
    return best possible score
  return best_possible_score - submitted_score

normalized_score(state):
  if no submitted result:
    return 0
  return submitted_score / best_possible_score

found_acceptable(state):
  return 1 if submitted meeting was acceptable else 0
```

The actual RL reward is `normalized_score`.

So:

```text
invalid or missing submission -> reward 0
valid but suboptimal submission -> reward between 0 and 1
best possible submission -> reward 1
```

## The Environment Class

`CalendarSchedulingEnv` wraps everything into a Verifiers `StatefulToolEnv`.

Pseudocode:

```text
CalendarSchedulingEnv(dataset, max_turns):
  create rubric:
    main reward = normalized_score
    extra metrics = calendar_reward, optimum_gap, found_acceptable

  initialize StatefulToolEnv with:
    dataset
    rubric
    max_turns
    system prompt

  register tools:
    check_attendee_calendar
    view_attendee_constraints
    check_room_availability
    check_score
    submit_window
```

The system prompt tells the model:

```text
inspect calendars and constraints
use score feedback
submit one final meeting
use UTC
do not submit unless check_score returned acceptable=true for that exact choice
```

## Hidden State Injection

Tool functions need access to the hidden task state. The model should not provide this state itself.

Pseudocode:

```text
update_tool_args(tool_name, tool_args, messages, state):
  automatically add state to tool_args
  return tool_args
```

This is why tool schemas hide the `state` argument from the model.

## Tool: Check Attendee Calendar

Pseudocode:

```text
check_attendee_calendar(attendee):
  find attendee by name

  if not found:
    return error

  return:
    attendee name
    city
    time zone offset
    submission timezone = UTC
    busy blocks in UTC
    busy blocks in attendee local time
    remaining turns
```

## Tool: View Attendee Constraints

Pseudocode:

```text
view_attendee_constraints(attendee="all"):
  if attendee is "all":
    select all attendees
  else:
    select matching attendee

  for each selected attendee:
    return:
      required?
      importance weight
      local hard window
      local preferred window
      UTC hints for hard/preferred windows
      soft costs:
        back-to-back penalty
        bad days
        bad-day penalty
        optional omission utility

  include remaining turns
```

## Tool: Check Room Availability

Pseudocode:

```text
check_room_availability(room="all"):
  if room is "all":
    select all rooms
  else:
    select matching room

  return:
    room name
    room capacity
    room busy blocks in UTC
    remaining turns
```

## Tool: Check Score

This lets the model test a candidate without final submission.

Pseudocode:

```text
check_score(day, start_time, room, attendees):
  result = score_choice(problem, candidate)
  add remaining_turns to result
  return result
```

This tool is useful because it gives exact feedback:

```text
acceptable?
score
violations
attendee utilities
normalized score
```

## Tool: Submit Window

This is the final-answer tool.

Pseudocode:

```text
submit_window(day, start_time, room, attendees):
  result = score_choice(problem, candidate)

  save in environment state:
    submitted_result = result
    submitted_choice = candidate

  create final environment response:
    "Final submission received..."
    acceptable?
    raw score
    best possible score
    normalized score

  return result
```

Once this happens, the rollout has a final submitted answer and the rubric can score it.

## Loading The Environment

`load_environment(...)` is the standard Prime/Verifiers entrypoint.

Pseudocode:

```text
load_environment(difficulty="medium", num_examples=25, seed=7, max_turns=None):
  validate difficulty

  if max_turns was not provided:
    use difficulty default

  dataset = build_dataset(num_examples, difficulty, seed, max_turns)

  return CalendarSchedulingEnv(dataset, max_turns)
```

This is what Prime calls when running evals or training.

## End-To-End Rollout Flow

Putting it all together:

```text
Prime calls load_environment(...)

Environment builds dataset:
  generate synthetic task
  solve task exactly
  hide full problem in info.problem
  expose public prompt to model

Model receives prompt and tools

Model calls tools:
  inspect constraints
  inspect calendars
  inspect rooms
  check candidate scores

Model calls submit_window(...)

submit_window:
  verifies candidate with score_choice
  stores submitted result

Rubric:
  computes normalized_score
  records extra metrics

Prime logs reward and rollout transcript
```

## Where The Verifier Lives

The verifier is not one single function, but the main pieces are:

```text
score_choice:
  verifies one submitted meeting and computes raw score

solve_problem:
  finds the best possible meeting by checking all choices

submit_window:
  applies score_choice to the model's final answer

normalized_score:
  converts submitted score into the RL reward
```

In short:

```text
submit_window stores the model answer
score_choice grades it
solve_problem provides the oracle best score
normalized_score turns that into reward
```
