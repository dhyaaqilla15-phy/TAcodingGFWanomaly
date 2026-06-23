# Workspace Execution Rule

- Codex must never execute model training, fitting, fine-tuning, or a runner
  stage that launches training in this repository.
- Codex may edit or create training scripts, inspect data and completed
  artifacts, validate syntax, and prepare exact commands.
- The user runs every training command. Codex must provide the command and
  wait for the user to report that it has completed before analyzing results
  or continuing to a dependent stage.
- Codex must not launch training in the foreground or background, including
  through wrapper scripts such as `run_*_tuning.py`.
