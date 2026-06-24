from __future__ import annotations

import argparse
from collections import defaultdict

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from calendar_agent import build_problem, minutes_to_time, score_choice


BUSY_STYLE = "bold white on red"
ROOM_STYLE = "bold white on blue"
BEST_STYLE = "bold black on green"
OPEN_STYLE = "dim"


def block_label(name: str, style: str) -> Text:
    text = Text("  ", style=style)
    text.append(f" {name}", style=style)
    return text


def render_grid(problem: dict) -> Table:
    table = Table(title="Calendar Problem", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Day", style="bold")
    table.add_column("Time UTC", style="cyan")
    for attendee in problem["attendees"]:
        label = attendee["name"] + ("*" if attendee["required"] else "")
        table.add_column(label, justify="center")
    for room in problem["rooms"]:
        table.add_column(f"Room {room['name']}", justify="center")

    best = problem["best_choice"]
    best_start = best["start_time"] if best else ""
    best_day = best["day"] if best else -1
    best_room = best["room"] if best else ""
    duration = problem["meeting_duration"]
    for day in range(problem["days"]):
        for start in range(problem["workday_start"], problem["workday_end"], 30):
            end = start + 30
            row = [str(day), f"{minutes_to_time(start)}-{minutes_to_time(end)}"]
            for attendee in problem["attendees"]:
                busy = any(block["day"] == day and start < block["end"] and end > block["start"] for block in attendee["busy"])
                is_best = (
                    day == best_day
                    and minutes_to_time(start) >= best_start
                    and start < int(best_start[:2]) * 60 + int(best_start[3:]) + duration
                    and attendee["name"] in (best.get("attendees") or [])
                ) if best else False
                if is_best:
                    row.append(block_label("best", BEST_STYLE))
                elif busy:
                    row.append(block_label("busy", BUSY_STYLE))
                else:
                    row.append(Text("open", style=OPEN_STYLE))
            for room in problem["rooms"]:
                busy = any(block["day"] == day and start < block["end"] and end > block["start"] for block in room["busy"])
                is_best = day == best_day and room["name"] == best_room and minutes_to_time(start) == best_start
                if is_best:
                    row.append(block_label("chosen", BEST_STYLE))
                elif busy:
                    row.append(block_label("held", ROOM_STYLE))
                else:
                    row.append(Text("open", style=OPEN_STYLE))
            table.add_row(*row)
    return table


def render_attendees(problem: dict) -> Table:
    table = Table(title="Attendees and Constraints", box=box.ROUNDED, expand=True)
    table.add_column("Name", style="bold")
    table.add_column("Role")
    table.add_column("Weight", justify="right")
    table.add_column("City")
    table.add_column("Hard Local")
    table.add_column("Preferred Local")
    table.add_column("Soft Notes")
    for attendee in problem["attendees"]:
        notes = []
        if attendee["bad_days"]:
            notes.append(f"dislikes days {attendee['bad_days']}")
        notes.append(f"B2B cost {attendee['back_to_back_cost']}")
        if not attendee["required"]:
            notes.append(f"omit utility {attendee['omit_utility']}")
        table.add_row(
            attendee["name"],
            "required" if attendee["required"] else "optional",
            f"{attendee['weight']:.3f}",
            attendee["city"],
            f"{minutes_to_time(attendee['hard_start'])}-{minutes_to_time(attendee['hard_end'])}",
            f"{minutes_to_time(attendee['pref_start'])}-{minutes_to_time(attendee['pref_end'])}",
            "; ".join(notes),
        )
    return table


def render_solution(problem: dict) -> Panel:
    best = problem["best_choice"]
    result = score_choice(problem, best) if best else {"score": 0.0, "attendee_scores": []}
    lines = [
        f"difficulty: {problem['difficulty']}",
        f"problem id: {problem['problem_id']}",
        f"best possible score: {problem['best_score']:.4f}",
        f"valid windows: {problem['valid_count']} / {problem['candidate_count']} ({problem['valid_ratio']:.2%})",
    ]
    if best:
        lines.append(f"best window: day {best['day']} at {best['start_time']} UTC in {best['room']}")
        lines.append(f"attendees: {', '.join(best['attendees'])}")
    lines.append("")
    lines.append("attendee utilities:")
    for item in result["attendee_scores"]:
        lines.append(f"  {item['name']}: {item['utility']:.3f} x {item['weight']:.3f}")
    return Panel("\n".join(lines), title="Solver Check", border_style="green")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a generated calendar scheduling problem.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    problem = build_problem(seed=args.seed, difficulty=args.difficulty)
    console = Console()
    layout = Layout()
    layout.split_column(Layout(name="top", size=12), Layout(name="calendar"), Layout(name="bottom", size=12))
    layout["top"].update(render_attendees(problem))
    layout["calendar"].update(render_grid(problem))
    layout["bottom"].update(render_solution(problem))
    console.print(layout)


if __name__ == "__main__":
    main()
