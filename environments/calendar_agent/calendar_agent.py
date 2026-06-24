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
TIME_ZONES = {
    "Los Angeles": -7 * 60,
    "Denver": -6 * 60,
    "Chicago": -5 * 60,
    "New York": -4 * 60,
    "London": 60,
}


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


def format_block(block: dict[str, Any], tz_offset: int = 0) -> dict[str, Any]:
    return {
        "day": block["day"],
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
    day = int(choice["day"])
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
                            "day": day,
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


def sample_attendee_names(rng: random.Random, count: int) -> list[str]:
    names = list(NAMES)
    rng.shuffle(names)
    return names[:count]


def sample_room_names(seed: int, attempt: int, count: int) -> list[str]:
    names = list(ROOM_NAMES)
    random.Random(seed * 9176 + attempt).shuffle(names)
    return names[:count]


def build_problem(seed: int, difficulty: str) -> dict[str, Any]:
    preset = DIFFICULTY_PRESETS[difficulty]
    for attempt in range(600):
        local_rng = random.Random(seed * 1009 + attempt)
        attendee_count = local_rng.randint(*preset["attendees"])
        optional_count = min(local_rng.randint(*preset["optional"]), attendee_count - 2)
        day_count = local_rng.randint(*preset["days"])
        duration = local_rng.choice([30, 60, 60, 90])
        room_count = local_rng.randint(*preset["rooms"])
        room_names = sample_room_names(seed, attempt, room_count)

        weights = [local_rng.uniform(0.7, 1.8) for _ in range(attendee_count)]
        weight_total = sum(weights)
        attendees = []
        tz_items = list(TIME_ZONES.items())
        attendee_names = sample_attendee_names(local_rng, attendee_count)
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
            "days": day_count,
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


def public_summary(problem: dict[str, Any]) -> str:
    required = [att["name"] for att in problem["attendees"] if att["required"]]
    optional = [att["name"] for att in problem["attendees"] if not att["required"]]
    room_names = ", ".join(room["name"] for room in problem["rooms"])
    return (
        f"Schedule a {problem['meeting_duration']}-minute meeting within days 0-{problem['days'] - 1}. "
        f"Use 30-minute start increments between {minutes_to_time(problem['workday_start'])} and "
        f"{minutes_to_time(problem['workday_end'])} UTC. Required attendees: {', '.join(required)}. "
        f"Optional desired attendees: {', '.join(optional) or 'none'}. Available rooms: {room_names}. "
        "Use the tools to inspect calendars and constraints, then submit one meeting window."
    )


def build_dataset(num_examples: int, difficulty: str, seed: int, max_turns: int) -> Dataset:
    rows = []
    for index in range(num_examples):
        problem = build_problem(seed + index, difficulty)
        problem["max_turns"] = max_turns
        rows.append(
            {
                "example_id": index,
                "question": public_summary(problem),
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


class CalendarSchedulingEnv(vf.StatefulToolEnv):
    def __init__(self, dataset: Dataset, max_turns: int):
        rubric = vf.Rubric(funcs=[normalized_score], weights=[1.0])
        rubric.add_metric(calendar_reward)
        rubric.add_metric(optimum_gap)
        rubric.add_metric(found_acceptable)
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
                "busy_blocks_utc": [format_block(block) for block in person["busy"]],
                "busy_blocks_local": [format_block(block, person["tz_offset"]) for block in person["busy"]],
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
                        "bad_days": person["bad_days"],
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
                        "busy_blocks_utc": [format_block(block) for block in item["busy"]],
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
    **kwargs: Any,
) -> vf.Environment:
    """Load the calendar scheduling environment."""
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTY_PRESETS)}")
    resolved_max_turns = max_turns or int(DIFFICULTY_PRESETS[difficulty]["max_turns"])
    dataset = build_dataset(num_examples=num_examples, difficulty=difficulty, seed=seed, max_turns=resolved_max_turns)
    return CalendarSchedulingEnv(dataset=dataset, max_turns=resolved_max_turns)


if __name__ == "__main__":
    sample = build_problem(seed=7, difficulty="medium")
    print(json.dumps({"summary": public_summary(sample), "solution": solve_problem(sample)}, indent=2))
