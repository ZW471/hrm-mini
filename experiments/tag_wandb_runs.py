"""Add (or remove) a tag on existing W&B runs whose name matches a regex, so they can be hidden.

    .venv/bin/python experiments/tag_wandb_runs.py --regex '^dfm_L' --tag experimental
    .venv/bin/python experiments/tag_wandb_runs.py --regex '^dfm_L' --tag experimental --remove
    .venv/bin/python experiments/tag_wandb_runs.py --regex '^dfm_L' --tag experimental --dry-run
    .venv/bin/python experiments/tag_wandb_runs.py --regex '^tuned_hrm full' --seed 1 --tag 'memory test'

`--seed` narrows a multi-seed name to one run (the seeds share a display name and differ only in
`config.seed`).

Needs credentials for the entity (`wandb login`); the default path is the one the training scripts log to.
"""
import argparse

import wandb

p = argparse.ArgumentParser()
p.add_argument("--path", default="zhiyuwang-university-of-cambridge/sudoku")
p.add_argument("--regex", default=None, help="Regex on the run display name")
p.add_argument("--group", default=None, help="Regex on the run group instead. train.py sets the group to "
               "the checkpoint dir name and it survives a display-name rename in the UI, so it is the "
               "reliable way to reach the run behind a given checkpoint")
p.add_argument("--tag", required=True)
p.add_argument("--seed", type=int, default=None, help="Only runs whose config.seed equals this")
p.add_argument("--remove", action="store_true")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

api = wandb.Api()
if (args.regex is None) == (args.group is None):
    p.error("give exactly one of --regex / --group")
filters = {"group": {"$regex": args.group}} if args.group else {"display_name": {"$regex": args.regex}}
if args.seed is not None:
    filters["config.seed"] = args.seed
runs = list(api.runs(args.path, filters=filters))
print(f"{len(runs)} run(s) match {'group' if args.group else 'name'} {(args.group or args.regex)!r}"
      + (f" seed={args.seed}" if args.seed is not None else "") + f" in {args.path}")
for r in runs:
    tags = set(r.tags)
    new = tags - {args.tag} if args.remove else tags | {args.tag}
    flag = "" if new == tags else ("-" if args.remove else "+")
    print(f"  {r.id}  {r.name:36s} group={r.group!r:38s} {r.state:9s} tags={sorted(tags)} {flag}{args.tag if flag else ''}")
    if new != tags and not args.dry_run:
        r.tags = sorted(new)
        r.update()
