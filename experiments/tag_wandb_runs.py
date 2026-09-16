"""Add (or remove) a tag on existing W&B runs whose name matches a regex, so they can be hidden.

    .venv/bin/python experiments/tag_wandb_runs.py --regex '^dfm_L' --tag experimental
    .venv/bin/python experiments/tag_wandb_runs.py --regex '^dfm_L' --tag experimental --remove
    .venv/bin/python experiments/tag_wandb_runs.py --regex '^dfm_L' --tag experimental --dry-run

Needs credentials for the entity (`wandb login`); the default path is the one the training scripts log to.
"""
import argparse

import wandb

p = argparse.ArgumentParser()
p.add_argument("--path", default="zhiyuwang-university-of-cambridge/sudoku")
p.add_argument("--regex", required=True, help="Regex on the run display name")
p.add_argument("--tag", required=True)
p.add_argument("--remove", action="store_true")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

api = wandb.Api()
runs = list(api.runs(args.path, filters={"display_name": {"$regex": args.regex}}))
print(f"{len(runs)} run(s) match {args.regex!r} in {args.path}")
for r in runs:
    tags = set(r.tags)
    new = tags - {args.tag} if args.remove else tags | {args.tag}
    flag = "" if new == tags else ("-" if args.remove else "+")
    print(f"  {r.id}  {r.name:48s} {r.state:9s} tags={sorted(tags)} {flag}{args.tag if flag else ''}")
    if new != tags and not args.dry_run:
        r.tags = sorted(new)
        r.update()
