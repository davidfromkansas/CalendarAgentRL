Make an environment for a calendar scheduling agent.
In each task, there should be a set of people with busy calendars, and individual + global constraints for scheduling the meeting.
Some constraints can be "hard" (not allowed to violate), others can be "soft", where violating a constraint incurs some utility cost for certain attendees.
Each attendee has a utility for the proposed meeting time between 0 and 1, and the task score will be the weighted average of attendee scores if an acceptable meeting time is found, and 0 otherwise.
Attendee importance weights should be normalized to 1 for each task.

We should be able to programmatically generate task problems, and deterministically validate that satisfying solutions exist (and what their best possible score would be).
We should have fine-grained controls for key degrees of freedom in task generation, with higher-level parameters ("easy" / "medium" / "hard") for the full task set, which then map into setting ranges for the more fine-grained controls.
Be creative, and use your judgment to design clean composition rules for converting meeting choices and conflicts into scores. Avoid complex branching/conditional logic where possible.
Think carefully about designing your system in a way which discourages "backdoor" strategies or reward hacks.
The best approach for an agent should be to make a good-faith effort to satisfy constraints as best as possible.
Experiment with sampling strategies to ensure that tasks are solvable most of the time (so that we can pre-filter any unsolvable tasks cheaply), and that they aren't too easy -- there shouldn't be an abundance of valid solutions, random proposal times should be a bad strategy.

Types of constraints we want to potentially account for:

- Conflicting schedules
- Time zones + early/late/day preferences
- Meeting length
- Room availability
- Back-to-back meeting preferences
- Desired-but-optional attendees
- Other related constraints which reflect real-world calendar challenges

Degrees of freedom:

- Number of attendees
- Window of consideration
- Types of constraints
- Tightness of constraints

Use the StatefulToolEnv pattern, and in-memory data structures for the calendar + attendee information. The agent should have tools for things like:

- Checking attendee calendars
- Viewing attendee constraints
- Checking score of a proposed window
- Submitting a window

The environment should have a max_turns parameter, and tool results should show the remaining turns to the agent.
Default limit should be enough to allow reasonable exploration, but not so high that the agent can brute-force search all times.

We should also have a nice standalone script in the environment which creates a TUI to visualize a "calendar problem" similar to typical meeting apps, including attendees, timeblocks, and constraints, but fully in the terminal, using Rich styling, similar design language to the `prime eval tui` viewer implemented within the `verifiers` library (inspect verifiers source for reference).

Create a detailed design doc and plan for testing (PLAN.md), implement in full, revise PLAN.md after major milestones to reflect accomplishments and updated TODOs, and run basic small evals throughout as needed.
You are welcome to use the PRIME_API_KEY set in my environment for inference tests (see configs/endpoints.toml for models).
Let me know when you're happy with your implementation.
