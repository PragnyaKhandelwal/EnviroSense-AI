#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/pragnya/EnviroSenseAI"
BRANCH="${DEPLOY_BRANCH:-Pragnya}"
REMOTE="origin"
GIT_USER="pragnya"

run_git() {
  sudo -u "$GIT_USER" -H bash -lc "cd '$REPO_DIR' && $*"
}

if [[ -n "$(run_git "git status --porcelain")" ]]; then
  echo "Repository has local changes; skipping auto-deploy to avoid conflicts."
  exit 0
fi

run_git "git fetch '$REMOTE' '$BRANCH'"
LOCAL_SHA="$(run_git "git rev-parse HEAD")"
REMOTE_SHA="$(run_git "git rev-parse '$REMOTE/$BRANCH'")"

if [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]]; then
  echo "No new commits for $BRANCH."
  exit 0
fi

run_git "git checkout '$BRANCH'"
run_git "git pull --ff-only '$REMOTE' '$BRANCH'"

# Restart long-running subscriber and trigger one Pragnya batch for smoke check.
systemctl restart subscriber.service
systemctl start pragnya-impute.service

echo "Deployed $BRANCH: $LOCAL_SHA -> $REMOTE_SHA"
