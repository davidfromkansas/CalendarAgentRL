from __future__ import annotations

import itertools
import json
import math
import random
from copy import deepcopy
from typing import Any

import verifiers as vf
from datasets import Dataset


UTC_DAY_START = 9 * 60
SLOT_MINUTES = 30
ROOM_NAMES = ["Atlas", "Borealis", "Cascade"]
GENERALIZATION_ROOM_NAMES = ["Juniper", "Kepler", "Mariner", "Solstice"]
NAMES = [
    "Avery",
    "Blair",
    "Casey",
    "Devon",
    "Emery",
    "Finley",
    "Harper",
    "Indigo",
    "Jordan",
    "Kai",
    "Logan",
    "Morgan",
    "Nova",
    "Parker",
    "Quinn",
    "Reese",
    "Riley",
    "Sage",
    "Taylor",
    "Uma",
    "Val",
    "Winter",
    "Xen",
    "Yael",
    "Zion",
]
GENERALIZATION_NAMES = [
    "Amara",
    "Bastian",
    "Cleo",
    "Daria",
    "Eli",
    "Farah",
    "Galen",
    "Hana",
    "Ilya",
    "Jules",
    "Kiran",
    "Leona",
    "Mika",
    "Noor",
    "Oren",
    "Priya",
    "Rafi",
    "Selah",
    "Toma",
    "Vera",
    "Wes",
    "Yara",
]
TIME_ZONES = {
    "Los Angeles": -7 * 60,
    "Denver": -6 * 60,
    "Chicago": -5 * 60,
    "New York": -4 * 60,
    "London": 60,
}
GENERALIZATION_TIME_ZONES = {
    **TIME_ZONES,
    "Toronto": -4 * 60,
    "Berlin": 2 * 60,
    "Lisbon": 60,
    "Singapore": 8 * 60,
}

SPLIT_PRESETS = {
    "train_easy": {"difficulty": "easy", "seed": 10000, "profile": "standard"},
    "dev_easy": {"difficulty": "easy", "seed": 20000, "profile": "standard"},
    "heldout_easy": {"difficulty": "easy", "seed": 30000, "profile": "standard"},
    "train_medium": {"difficulty": "medium", "seed": 40000, "profile": "standard"},
    "dev_medium": {"difficulty": "medium", "seed": 50000, "profile": "standard"},
    "heldout_medium": {"difficulty": "medium", "seed": 60000, "profile": "standard"},
    "heldout_generalization": {"difficulty": "medium", "seed": 90000, "profile": "generalization"},
}

PROMPT_VARIANTS = ("default", "brief", "ticket", "stakeholder")


DIFFICULTY_PRESETS = {
    "easy": {
        "attendees": (3, 4),
        "days": (3, 4),
        "busy_blocks": (2, 4),
        "optional": (1, 1),
        "rooms": (2, 3),
        "target_valid_ratio": (0.05, 0.28),
        "min_best_score": 0.72,
        "max_turns": 12,
    },
    "medium": {
        "attendees": (4, 6),
        "days": (2, 3),
        "busy_blocks": (4, 6),
        "optional": (1, 2),
        "rooms": (2, 3),
        "target_valid_ratio": (0.02, 0.16),
        "min_best_score": 0.62,
        "max_turns": 14,
    },
    "hard": {
        "attendees": (6, 8),
        "days": (2, 3),
        "busy_blocks": (5, 8),
        "optional": (2, 3),
        "rooms": (1, 2),
        "target_valid_ratio": (0.006, 0.08),
        "min_best_score": 0.50,
        "max_turns": 16,
    },
}


def minutes_to_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def time_to_minutes(value: str) -> int:
    hour, minute = value.strip().split(":", 1)
    parsed = int(hour) * 60 + int(minute)
    if parsed % SLOT_MINUTES != 0:
        raise ValueError("Times must land on 30-minute boundaries.")
    return parsed


def overlaps(start: int, end: int, block: dict[str, Any]) -> bool:
    return start < int(block["end"]) and end > int(block["start"])


def local_minutes(utc_minute: int, tz_offset: int) -> int:
    return (utc_minute + tz_offset) % (24 * 60)


def display_day(problem: dict[str, Any], internal_day: int) -> int:
    day_labels = problem.get("day_labels")
    if day_labels:
        return int(day_labels[internal_day])
    return internal_day


def internal_day(problem: dict[str, Any], submitted_day: int) -> int:
    day_labels = [int(day) for day in problem.get("day_labels", [])]
    if submitted_day in day_labels:
        return day_labels.index(submitted_day)
    if not day_labels and 0 <= submitted_day < int(problem["days"]):
        return submitted_day
    raise ValueError(f"day must be one of {day_labels or list(range(int(problem['days'])))}")


def format_block(block: dict[str, Any], tz_offset: int = 0, problem: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "day": display_day(problem, int(block["day"])) if problem else block["day"],
        "start": minutes_to_time(local_minutes(int(block["start"]), tz_offset)),
        "end": minutes_to_time(local_minutes(int(block["end"]), tz_offset)),
    }


def attendee_utility(attendee: dict[str, Any], day: int, start: int, end: int, included: bool) -> float:
    if not included:
        return 0.0 if attendee["required"] else attendee["omit_utility"]
    if any(block["day"] == day and overlaps(start, end, block) for block in attendee["busy"]):
        return 0.0
    local_start = local_minutes(start, attendee["tz_offset"])
    local_end = local_minutes(end, attendee["tz_offset"])
    if local_start < attendee["hard_start"] or local_end > attendee["hard_end"]:
        return 0.0

    pref_center = (attendee["pref_start"] + attendee["pref_end"]) / 2
    meeting_center = (local_start + local_end) / 2
    pref_span = max(1.0, (attendee["pref_end"] - attendee["pref_start"]) / 2)
    preference_cost = min(0.38, 0.16 * abs(meeting_center - pref_center) / pref_span)

    adjacent = any(
        block["day"] == day and (abs(start - int(block["end"])) <= 15 or abs(end - int(block["start"])) <= 15)
        for block in attendee["busy"]
    )
    back_to_back_cost = attendee["back_to_back_cost"] if adjacent else 0.0
    day_cost = attendee["bad_day_cost"] if day in attendee["bad_days"] else 0.0
    return round(max(0.0, 1.0 - preference_cost - back_to_back_cost - day_cost), 4)


def score_choice(problem: dict[str, Any], choice: dict[str, Any]) -> dict[str, Any]:
    day = internal_day(problem, int(choice["day"]))
    start = time_to_minutes(choice["start_time"]) if isinstance(choice["start_time"], str) else int(choice["start_time"])
    duration = int(problem["meeting_duration"])
    end = start + duration
    room = str(choice["room"])
    included = set(choice.get("attendees") or [att["name"] for att in problem["attendees"]])
    violations: list[str] = []

    if day < 0 or day >= int(problem["days"]):
        violations.append("day outside scheduling window")
    if start < int(problem["workday_start"]) or end > int(problem["workday_end"]):
        violations.append("meeting outside global working hours")
    if room not in {item["name"] for item in problem["rooms"]}:
        violations.append(f"unknown room: {room}")
    else:
        room_data = next(item for item in problem["rooms"] if item["name"] == room)
        if any(block["day"] == day and overlaps(start, end, block) for block in room_data["busy"]):
            violations.append(f"room {room} is unavailable")

    attendee_scores = []
    for attendee in problem["attendees"]:
        is_included = attendee["name"] in included
        utility = attendee_utility(attendee, day, start, end, is_included)
        if attendee["required"] and not is_included:
            violations.append(f"required attendee {attendee['name']} omitted")
        if attendee["required"] and is_included and utility <= 0:
            violations.append(f"required attendee {attendee['name']} has a hard conflict")
        attendee_scores.append(
            {
                "name": attendee["name"],
                "included": is_included,
                "required": attendee["required"],
                "weight": attendee["weight"],
                "utility": utility,
                "weighted_utility": round(attendee["weight"] * utility, 4),
            }
        )

    acceptable = not violations
    score = round(sum(item["weighted_utility"] for item in attendee_scores), 4) if acceptable else 0.0
    return {
        "acceptable": acceptable,
        "score": score,
        "violations": violations,
        "attendee_scores": attendee_scores,
        "normalized_against_optimum": round(score / max(problem["best_score"], 1e-9), 4)
        if problem.get("best_score")
        else 0.0,
    }


def enumerate_choices(problem: dict[str, Any]) -> list[dict[str, Any]]:
    required = [att["name"] for att in problem["attendees"] if att["required"]]
    optional = [att["name"] for att in problem["attendees"] if not att["required"]]
    subsets = []
    for size in range(len(optional) + 1):
        for extra in itertools.combinations(optional, size):
            subsets.append(required + list(extra))
    choices = []
    for day in range(int(problem["days"])):
        for start in range(int(problem["workday_start"]), int(problem["workday_end"]) - int(problem["meeting_duration"]) + 1, SLOT_MINUTES):
            for room in problem["rooms"]:
                for attendees in subsets:
                    choices.append(
                        {
                            "day": display_day(problem, day),
                            "start_time": minutes_to_time(start),
                            "room": room["name"],
                            "attendees": attendees,
                        }
                    )
    return choices


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    best_choice: dict[str, Any] | None = None
    best_result: dict[str, Any] | None = None
    valid_count = 0
    all_choices = enumerate_choices(problem)
    for choice in all_choices:
        result = score_choice({**problem, "best_score": 0.0}, choice)
        if result["acceptable"]:
            valid_count += 1
        if best_result is None or result["score"] > best_result["score"]:
            best_choice = choice
            best_result = result
    return {
        "best_choice": best_choice,
        "best_score": 0.0 if best_result is None else best_result["score"],
        "valid_count": valid_count,
        "candidate_count": len(all_choices),
        "valid_ratio": round(valid_count / max(len(all_choices), 1), 4),
    }


def random_block(rng: random.Random, day_count: int, min_start: int, max_end: int) -> dict[str, Any]:
    start = rng.randrange(min_start, max_end - 30, SLOT_MINUTES)
    duration = rng.choice([30, 60, 60, 90, 120])
    end = min(max_end, start + duration)
    return {"day": rng.randrange(day_count), "start": start, "end": end}


def sample_attendee_names(rng: random.Random, count: int, profile: str) -> list[str]:
    names = list(GENERALIZATION_NAMES if profile == "generalization" else NAMES)
    rng.shuffle(names)
    return names[:count]


def sample_room_names(seed: int, attempt: int, count: int, profile: str) -> list[str]:
    names = list(GENERALIZATION_ROOM_NAMES if profile == "generalization" else ROOM_NAMES)
    random.Random(seed * 9176 + attempt).shuffle(names)
    return names[:count]


def build_problem(seed: int, difficulty: str, profile: str = "standard") -> dict[str, Any]:
    preset = DIFFICULTY_PRESETS[difficulty]
    if profile not in {"standard", "generalization"}:
        raise ValueError("profile must be 'standard' or 'generalization'")
    for attempt in range(600):
        local_rng = random.Random(seed * 1009 + attempt)
        attendee_count = local_rng.randint(*preset["attendees"])
        optional_count = min(local_rng.randint(*preset["optional"]), attendee_count - 2)
        day_count = local_rng.randint(*preset["days"])
        day_start = local_rng.randint(1, 31 - day_count)
        day_labels = list(range(day_start, day_start + day_count))
        duration = local_rng.choice([30, 60, 60, 90])
        room_count = local_rng.randint(*preset["rooms"])
        room_names = sample_room_names(seed, attempt, room_count, profile)

        weights = [local_rng.uniform(0.7, 1.8) for _ in range(attendee_count)]
        weight_total = sum(weights)
        attendees = []
        tz_items = list((GENERALIZATION_TIME_ZONES if profile == "generalization" else TIME_ZONES).items())
        attendee_names = sample_attendee_names(local_rng, attendee_count, profile)
        for index, name in enumerate(attendee_names):
            city, offset = local_rng.choice(tz_items)
            hard_start = local_rng.choice([7 * 60, 8 * 60, 9 * 60])
            hard_end = local_rng.choice([17 * 60, 18 * 60, 19 * 60])
            pref_start = local_rng.choice([9 * 60, 10 * 60, 11 * 60, 13 * 60])
            pref_end = min(hard_end, pref_start + local_rng.choice([180, 240, 300]))
            busy = [
                random_block(local_rng, day_count, UTC_DAY_START, 17 * 60)
                for _ in range(local_rng.randint(*preset["busy_blocks"]))
            ]
            attendees.append(
                {
                    "name": name,
                    "city": city,
                    "tz_offset": offset,
                    "required": index < attendee_count - optional_count,
                    "weight": round(weights[index] / weight_total, 4),
                    "hard_start": hard_start,
                    "hard_end": hard_end,
                    "pref_start": pref_start,
                    "pref_end": pref_end,
                    "back_to_back_cost": round(local_rng.uniform(0.06, 0.2), 3),
                    "bad_days": sorted(local_rng.sample(range(day_count), k=local_rng.randint(0, min(1, day_count)))),
                    "bad_day_cost": round(local_rng.uniform(0.05, 0.16), 3),
                    "omit_utility": round(local_rng.uniform(0.08, 0.28), 3),
                    "busy": busy,
                }
            )
        total_weight = sum(att["weight"] for att in attendees)
        attendees[-1]["weight"] = round(attendees[-1]["weight"] + (1.0 - total_weight), 4)

        rooms = [
            {
                "name": room_name,
                "capacity": attendee_count + local_rng.randint(0, 3),
                "busy": [
                    random_block(local_rng, day_count, UTC_DAY_START, 17 * 60)
                    for _ in range(local_rng.randint(1, max(2, preset["busy_blocks"][1] // 2)))
                ],
            }
            for room_name in room_names
        ]
        problem = {
            "problem_id": f"calendar-{difficulty}-{seed}-{attempt}",
            "difficulty": difficulty,
            "generation_profile": profile,
            "days": day_count,
            "day_labels": day_labels,
            "meeting_duration": duration,
            "workday_start": UTC_DAY_START,
            "workday_end": 17 * 60,
            "attendees": attendees,
            "rooms": rooms,
        }
        solution = solve_problem(problem)
        low, high = preset["target_valid_ratio"]
        if solution["best_choice"] and solution["best_score"] >= preset["min_best_score"] and low <= solution["valid_ratio"] <= high:
            problem.update(solution)
            return problem
    problem.update(solution)
    return problem


def slice_tags(problem: dict[str, Any], prompt_variant: str = "default", split: str | None = None) -> dict[str, Any]:
    optional_count = sum(1 for attendee in problem["attendees"] if not attendee["required"])
    required_count = len(problem["attendees"]) - optional_count
    timezone_offsets = [int(attendee["tz_offset"]) for attendee in problem["attendees"]]
    timezone_span_hours = round((max(timezone_offsets) - min(timezone_offsets)) / 60, 1) if timezone_offsets else 0.0
    valid_ratio = float(problem["valid_ratio"])
    best_start = time_to_minutes(problem["best_choice"]["start_time"]) if problem.get("best_choice") else 0
    if valid_ratio < 0.04:
        density_bucket = "low"
    elif valid_ratio < 0.10:
        density_bucket = "medium"
    else:
        density_bucket = "high"
    if best_start < 12 * 60:
        best_start_bucket = "morning"
    elif best_start < 15 * 60:
        best_start_bucket = "midday"
    else:
        best_start_bucket = "late"
    return {
        "split": split or "custom",
        "difficulty": problem["difficulty"],
        "generation_profile": problem.get("generation_profile", "standard"),
        "prompt_variant": prompt_variant,
        "attendee_count": len(problem["attendees"]),
        "required_count": required_count,
        "optional_count": optional_count,
        "room_count": len(problem["rooms"]),
        "day_count": int(problem["days"]),
        "day_start": int(problem.get("day_labels", [0])[0]),
        "day_end": int(problem.get("day_labels", [int(problem["days"]) - 1])[-1]),
        "duration_minutes": int(problem["meeting_duration"]),
        "timezone_span_hours": timezone_span_hours,
        "valid_ratio": valid_ratio,
        "valid_density_bucket": density_bucket,
        "best_start_bucket": best_start_bucket,
        "best_score": float(problem["best_score"]),
    }


def select_prompt_variant(seed: int, index: int, prompt_variant: str) -> str:
    if prompt_variant == "mixed":
        return PROMPT_VARIANTS[(seed + index) % len(PROMPT_VARIANTS)]
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"prompt_variant must be 'mixed' or one of {PROMPT_VARIANTS}")
    return prompt_variant


def public_summary(problem: dict[str, Any], variant: str = "default") -> str:
    required = [att["name"] for att in problem["attendees"] if att["required"]]
    optional = [att["name"] for att in problem["attendees"] if not att["required"]]
    room_names = ", ".join(room["name"] for room in problem["rooms"])
    duration = problem["meeting_duration"]
    day_labels = [int(day) for day in problem.get("day_labels", list(range(int(problem["days"]))))]
    days = f"{day_labels[0]}-{day_labels[-1]}" if len(day_labels) > 1 else str(day_labels[0])
    window = f"{minutes_to_time(problem['workday_start'])}-{minutes_to_time(problem['workday_end'])} UTC"
    if variant == "brief":
        return (
            f"Find the best feasible slot for a {duration}-minute meeting. Days: {days}. "
            f"Starts must be every 30 minutes in the global window {window}. Must include: {', '.join(required)}. "
            f"Include optional people when useful: {', '.join(optional) or 'none'}. Rooms to consider: {room_names}. "
            "Inspect calendars, constraints, and rooms with tools; submit the chosen window."
        )
    if variant == "ticket":
        return (
            "Scheduling ticket:\n"
            f"- Meeting length: {duration} minutes\n"
            f"- Candidate days: {days}\n"
            f"- Allowed UTC start window: {window}, 30-minute increments\n"
            f"- Required participants: {', '.join(required)}\n"
            f"- Nice-to-have participants: {', '.join(optional) or 'none'}\n"
            f"- Rooms: {room_names}\n"
            "Resolve the ticket by using the available tools and submitting one final meeting window."
        )
    if variant == "stakeholder":
        optional_clause = f"Try to include {', '.join(optional)} if the tradeoff is worthwhile. " if optional else ""
        return (
            f"A team needs one {duration}-minute meeting sometime on days {days}. "
            f"The global scheduling window is {window}, with starts only on half-hour marks. "
            f"The meeting cannot happen without {', '.join(required)}. "
            f"{optional_clause}"
            f"Candidate rooms are {room_names}. Use the tools to discover hidden calendars/preferences and submit one window."
        )
    return (
        f"Schedule a {duration}-minute meeting within days {days}. "
        f"Use 30-minute start increments between {window}. Required attendees: {', '.join(required)}. "
        f"Optional desired attendees: {', '.join(optional) or 'none'}. Available rooms: {room_names}. "
        "Use the tools to inspect calendars and constraints, then submit one meeting window."
    )


def build_dataset(
    num_examples: int,
    difficulty: str,
    seed: int,
    max_turns: int,
    split: str | None = None,
    profile: str = "standard",
    prompt_variant: str = "default",
) -> Dataset:
    rows = []
    for index in range(num_examples):
        problem = build_problem(seed + index, difficulty, profile=profile)
        problem["max_turns"] = max_turns
        variant = select_prompt_variant(seed, index, prompt_variant)
        tags = slice_tags(problem, prompt_variant=variant, split=split)
        problem["slice_tags"] = tags
        rows.append(
            {
                "example_id": index,
                "split": split or "custom",
                "generation_profile": profile,
                "prompt_variant": variant,
                "task_slices": json.dumps(tags, sort_keys=True),
                "question": public_summary(problem, variant=variant),
                "answer": json.dumps(
                    {
                        "best_choice": problem["best_choice"],
                        "best_score": problem["best_score"],
                        "valid_ratio": problem["valid_ratio"],
                    },
                    sort_keys=True,
                ),
                "info": {"problem": problem},
            }
        )
    return Dataset.from_list(rows)


def calendar_reward(state: vf.State) -> float:
    submitted = state.get("submitted_result")
    if not isinstance(submitted, dict):
        return 0.0
    return float(submitted.get("score", 0.0))


def optimum_gap(state: vf.State) -> float:
    submitted = state.get("submitted_result")
    problem = state["info"]["problem"]
    if not isinstance(submitted, dict):
        return float(problem["best_score"])
    return round(max(0.0, float(problem["best_score"]) - float(submitted.get("score", 0.0))), 4)


def normalized_score(state: vf.State) -> float:
    submitted = state.get("submitted_result")
    problem = state["info"]["problem"]
    if not isinstance(submitted, dict):
        return 0.0
    return round(float(submitted.get("score", 0.0)) / max(float(problem["best_score"]), 1e-9), 4)


def found_acceptable(state: vf.State) -> float:
    submitted = state.get("submitted_result")
    return float(isinstance(submitted, dict) and bool(submitted.get("acceptable")))


def exact_optimal(state: vf.State) -> float:
    return float(normalized_score(state) >= 0.999)


def submitted_any(state: vf.State) -> float:
    return float(isinstance(state.get("submitted_result"), dict))


def invalid_submission(state: vf.State) -> float:
    submitted = state.get("submitted_result")
    return float(isinstance(submitted, dict) and not bool(submitted.get("acceptable")))


def task_attendee_count(state: vf.State) -> float:
    return float(state["info"]["problem"]["slice_tags"]["attendee_count"])


def task_optional_count(state: vf.State) -> float:
    return float(state["info"]["problem"]["slice_tags"]["optional_count"])


def task_room_count(state: vf.State) -> float:
    return float(state["info"]["problem"]["slice_tags"]["room_count"])


def task_valid_ratio(state: vf.State) -> float:
    return float(state["info"]["problem"]["slice_tags"]["valid_ratio"])


def task_timezone_span_hours(state: vf.State) -> float:
    return float(state["info"]["problem"]["slice_tags"]["timezone_span_hours"])


def slice_low_valid_density(state: vf.State) -> float:
    return float(state["info"]["problem"]["slice_tags"]["valid_density_bucket"] == "low")


def slice_late_optimum(state: vf.State) -> float:
    return float(state["info"]["problem"]["slice_tags"]["best_start_bucket"] == "late")


class CalendarSchedulingEnv(vf.StatefulToolEnv):
    def __init__(self, dataset: Dataset, max_turns: int):
        rubric = vf.Rubric(funcs=[normalized_score], weights=[1.0])
        rubric.add_metric(calendar_reward)
        rubric.add_metric(optimum_gap)
        rubric.add_metric(found_acceptable)
        rubric.add_metric(exact_optimal)
        rubric.add_metric(submitted_any)
        rubric.add_metric(invalid_submission)
        rubric.add_metric(task_attendee_count)
        rubric.add_metric(task_optional_count)
        rubric.add_metric(task_room_count)
        rubric.add_metric(task_valid_ratio)
        rubric.add_metric(task_timezone_span_hours)
        rubric.add_metric(slice_low_valid_density)
        rubric.add_metric(slice_late_optimum)
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            max_turns=max_turns,
            system_prompt=(
                "You are a careful calendar scheduling agent. Inspect the available calendars, "
                "constraints, rooms, and score feedback with tools. Submit exactly one final "
                "meeting window when you have a strong candidate. Keep reasoning concise: after "
                "checking constraints, calendars, and rooms, use check_score on one or more "
                "promising candidates and submit before the turn limit. All submitted start times "
                "must be UTC. Attendee calendar tools provide UTC blocks; use those for conflict "
                "checks. Never call submit_window unless check_score returned acceptable=true for "
                "that exact day, start_time, room, and attendee list. Do not guess hidden data."
            ),
        )
        for tool in [
            self.check_attendee_calendar,
            self.view_attendee_constraints,
            self.check_room_availability,
            self.check_score,
            self.submit_window,
        ]:
            self.add_tool(tool, args_to_skip=["state"])

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tool_args["state"] = state
        return tool_args

    def _remaining_turns(self, state: vf.State) -> int:
        return max(0, self.max_turns - len(state["trajectory"]) - 1)

    def _problem(self, state: vf.State) -> dict[str, Any]:
        return state["info"]["problem"]

    def check_attendee_calendar(self, attendee: str, state: vf.State) -> str:
        """View one attendee's busy calendar blocks in UTC and their local time."""
        problem = self._problem(state)
        person = next((item for item in problem["attendees"] if item["name"].lower() == attendee.lower()), None)
        if person is None:
            return json.dumps({"error": f"unknown attendee {attendee}", "remaining_turns": self._remaining_turns(state)})
        return json.dumps(
            {
                "attendee": person["name"],
                "city": person["city"],
                "timezone_offset_minutes": person["tz_offset"],
                "submission_timezone": "UTC",
                "busy_blocks_utc": [format_block(block, problem=problem) for block in person["busy"]],
                "busy_blocks_local": [format_block(block, person["tz_offset"], problem=problem) for block in person["busy"]],
                "remaining_turns": self._remaining_turns(state),
            },
            indent=2,
        )

    def view_attendee_constraints(self, attendee: str = "all", state: vf.State | None = None) -> str:
        """View hard and soft constraints for one attendee or all attendees."""
        assert state is not None
        problem = self._problem(state)
        selected = problem["attendees"]
        if attendee.lower() != "all":
            selected = [item for item in selected if item["name"].lower() == attendee.lower()]
        if not selected:
            return json.dumps({"error": f"unknown attendee {attendee}", "remaining_turns": self._remaining_turns(state)})
        constraints = []
        for person in selected:
            constraints.append(
                {
                    "name": person["name"],
                    "required": person["required"],
                    "importance_weight": person["weight"],
                    "local_hard_window": {
                        "start": minutes_to_time(person["hard_start"]),
                        "end": minutes_to_time(person["hard_end"]),
                    },
                    "local_preferred_window": {
                        "start": minutes_to_time(person["pref_start"]),
                        "end": minutes_to_time(person["pref_end"]),
                    },
                    "utc_hard_window_hint": {
                        "start": minutes_to_time(local_minutes(person["hard_start"], -person["tz_offset"])),
                        "end": minutes_to_time(local_minutes(person["hard_end"], -person["tz_offset"])),
                        "note": "Use check_score for final validation when local windows wrap UTC midnight.",
                    },
                    "utc_preferred_window_hint": {
                        "start": minutes_to_time(local_minutes(person["pref_start"], -person["tz_offset"])),
                        "end": minutes_to_time(local_minutes(person["pref_end"], -person["tz_offset"])),
                        "note": "Soft preference hint only; check_score computes exact utility.",
                    },
                    "soft_costs": {
                        "back_to_back": person["back_to_back_cost"],
                        "bad_days": [display_day(problem, int(day)) for day in person["bad_days"]],
                        "bad_day_cost": person["bad_day_cost"],
                        "optional_omission_utility": person["omit_utility"] if not person["required"] else None,
                    },
                }
            )
        return json.dumps({"constraints": constraints, "remaining_turns": self._remaining_turns(state)}, indent=2)

    def check_room_availability(self, room: str = "all", state: vf.State | None = None) -> str:
        """View busy blocks for one room or all rooms in UTC."""
        assert state is not None
        problem = self._problem(state)
        rooms = problem["rooms"]
        if room.lower() != "all":
            rooms = [item for item in rooms if item["name"].lower() == room.lower()]
        if not rooms:
            return json.dumps({"error": f"unknown room {room}", "remaining_turns": self._remaining_turns(state)})
        return json.dumps(
            {
                "rooms": [
                    {
                        "name": item["name"],
                        "capacity": item["capacity"],
                        "busy_blocks_utc": [format_block(block, problem=problem) for block in item["busy"]],
                    }
                    for item in rooms
                ],
                "remaining_turns": self._remaining_turns(state),
            },
            indent=2,
        )

    def check_score(
        self,
        day: int,
        start_time: str,
        room: str,
        attendees: list[str] | None = None,
        state: vf.State | None = None,
    ) -> str:
        """Score a candidate meeting without submitting it."""
        assert state is not None
        problem = self._problem(state)
        try:
            result = score_choice(problem, {"day": day, "start_time": start_time, "room": room, "attendees": attendees})
        except Exception as exc:
            return json.dumps({"error": str(exc), "remaining_turns": self._remaining_turns(state)})
        result["remaining_turns"] = self._remaining_turns(state)
        return json.dumps(result, indent=2)

    def submit_window(
        self,
        day: int,
        start_time: str,
        room: str,
        attendees: list[str] | None = None,
        state: vf.State | None = None,
    ) -> str:
        """Submit the final meeting window for scoring."""
        assert state is not None
        problem = self._problem(state)
        try:
            result = score_choice(problem, {"day": day, "start_time": start_time, "room": room, "attendees": attendees})
        except Exception as exc:
            result = {"acceptable": False, "score": 0.0, "violations": [str(exc)], "attendee_scores": []}
        state["submitted_result"] = result
        state["submitted_choice"] = {"day": day, "start_time": start_time, "room": room, "attendees": attendees}
        state["final_env_response"] = [
            vf.UserMessage(
                content=(
                    "Final submission received. "
                    f"Acceptable={result['acceptable']}; score={result['score']:.4f}; "
                    f"best_possible={problem['best_score']:.4f}; "
                    f"normalized_score={result.get('normalized_against_optimum', 0.0):.4f}."
                )
            )
        ]
        result["remaining_turns"] = self._remaining_turns(state)
        return json.dumps(result, indent=2)


def load_environment(
    difficulty: str = "medium",
    num_examples: int = 25,
    seed: int = 7,
    max_turns: int | None = None,
    split: str | None = None,
    generation_profile: str = "standard",
    prompt_variant: str = "default",
    **kwargs: Any,
) -> vf.Environment:
    """Load the calendar scheduling environment."""
    if split is not None:
        if split not in SPLIT_PRESETS:
            raise ValueError(f"split must be one of {sorted(SPLIT_PRESETS)}")
        preset = SPLIT_PRESETS[split]
        difficulty = str(preset["difficulty"])
        seed = int(preset["seed"])
        generation_profile = str(preset["profile"])
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTY_PRESETS)}")
    resolved_max_turns = max_turns or int(DIFFICULTY_PRESETS[difficulty]["max_turns"])
    dataset = build_dataset(
        num_examples=num_examples,
        difficulty=difficulty,
        seed=seed,
        max_turns=resolved_max_turns,
        split=split,
        profile=generation_profile,
        prompt_variant=prompt_variant,
    )
    return CalendarSchedulingEnv(dataset=dataset, max_turns=resolved_max_turns)


if __name__ == "__main__":
    sample = build_problem(seed=7, difficulty="medium")
    print(json.dumps({"summary": public_summary(sample), "solution": solve_problem(sample)}, indent=2))
