from __future__ import annotations

import json

from calendar_agent import build_problem, calendar_reward, load_environment, normalized_score, public_summary, score_choice


def test_generated_problems_are_solvable() -> None:
    for difficulty in ["easy", "medium", "hard"]:
        for seed in range(7, 10):
            problem = build_problem(seed=seed, difficulty=difficulty)
            assert problem["best_choice"] is not None
            assert problem["best_score"] > 0
            assert problem["valid_count"] > 0
            assert score_choice(problem, problem["best_choice"])["score"] == problem["best_score"]


def test_attendee_weights_are_normalized() -> None:
    problem = build_problem(seed=13, difficulty="medium")
    assert abs(sum(attendee["weight"] for attendee in problem["attendees"]) - 1.0) < 0.002


def test_attendee_names_vary_by_generated_task() -> None:
    first = build_problem(seed=401, difficulty="easy")
    first_again = build_problem(seed=401, difficulty="easy")
    second = build_problem(seed=402, difficulty="easy")

    first_names = [attendee["name"] for attendee in first["attendees"]]
    assert first_names == [attendee["name"] for attendee in first_again["attendees"]]
    assert first_names != [attendee["name"] for attendee in second["attendees"]]


def test_room_order_varies_by_generated_task() -> None:
    room_orders = {
        tuple(room["name"] for room in build_problem(seed=seed, difficulty="easy")["rooms"])
        for seed in range(401, 421)
    }
    assert len(room_orders) > 1


def test_visible_days_are_month_style_labels() -> None:
    problem = build_problem(seed=401, difficulty="easy")
    assert problem["day_labels"][0] >= 1
    assert problem["day_labels"][-1] <= 30
    assert problem["best_choice"]["day"] in problem["day_labels"]
    assert "days 0-" not in public_summary(problem)


def test_heldout_generalization_uses_distinct_surface_data() -> None:
    env = load_environment(split="heldout_generalization", num_examples=3, prompt_variant="mixed")
    dataset = env.get_dataset()
    profiles = {row["generation_profile"] for row in dataset}
    variants = {row["prompt_variant"] for row in dataset}
    assert profiles == {"generalization"}
    assert len(variants) > 1
    for row in dataset:
        problem = row["info"]["problem"]
        assert problem["generation_profile"] == "generalization"
        assert problem["slice_tags"]["split"] == "heldout_generalization"
        assert all(room["name"] not in {"Atlas", "Borealis", "Cascade"} for room in problem["rooms"])


def test_task_slice_columns_are_available() -> None:
    env = load_environment(difficulty="medium", num_examples=1, seed=500, prompt_variant="ticket")
    row = env.get_dataset()[0]
    slices = json.loads(row["task_slices"])
    assert row["prompt_variant"] == "ticket"
    assert slices["difficulty"] == "medium"
    assert slices["attendee_count"] >= slices["required_count"]
    assert slices["valid_density_bucket"] in {"low", "medium", "high"}


def test_environment_loads_and_hides_state_argument() -> None:
    env = load_environment(difficulty="easy", num_examples=1, seed=7)
    dataset = env.get_dataset()
    assert len(dataset) == 1
    assert json.loads(dataset[0]["answer"])["best_score"] > 0
    for tool_def in env.tool_defs:
        assert "state" not in tool_def.parameters.get("properties", {})


def test_training_reward_is_normalized_score() -> None:
    problem = build_problem(seed=401, difficulty="easy")
    result = score_choice(problem, problem["best_choice"])
    state = {"info": {"problem": problem}, "submitted_result": result}

    assert calendar_reward(state) == problem["best_score"]
    assert normalized_score(state) == 1.0


if __name__ == "__main__":
    test_generated_problems_are_solvable()
    test_attendee_weights_are_normalized()
    test_attendee_names_vary_by_generated_task()
    test_room_order_varies_by_generated_task()
    test_visible_days_are_month_style_labels()
    test_heldout_generalization_uses_distinct_surface_data()
    test_task_slice_columns_are_available()
    test_environment_loads_and_hides_state_argument()
    test_training_reward_is_normalized_score()
    print("calendar-agent tests passed")
