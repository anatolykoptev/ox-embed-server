#!/usr/bin/env bash
# Single-stage CI runner: prints live progress, tees to a per-stage log,
# reports elapsed time, and stops the chain on first failure.
#
# Usage: scripts/ci-stage.sh <name> <cmd> [args...]
#   $ scripts/ci-stage.sh fmt cargo fmt --all -- --check
#
# Pattern is the same shape dozor uses for compose deploys: every phase
# writes to its own log under $LOGDIR (default /tmp/embed-server-ci) and
# the operator can post-mortem any failed stage by `cat`-ing that file.
set -o pipefail

NAME="${1:?stage name required}"
shift

LOGDIR="${EMBED_CI_LOGDIR:-/tmp/embed-server-ci}"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="$LOGDIR/$TS-$NAME.log"

# Keep a stable "last-of-each-stage" symlink so `make logs` is trivial.
LAST_LINK="$LOGDIR/last-$NAME.log"
ln -sfn "$LOG" "$LAST_LINK"

# Keep colour escapes through `tee`.  Cargo emits per-crate `Compiling X v..`
# lines on stderr by default; we don't need the animated progress bar.
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"

START=$SECONDS
printf '\033[1;36m▶  %-12s\033[0m  →  %s\n' "$NAME" "$LOG"

# tee streams to terminal AND log; pipefail propagates inner failure.
if "$@" 2>&1 | tee "$LOG"; then
  DUR=$((SECONDS - START))
  printf '\033[1;32m✅ %-12s\033[0m  %ds  →  %s\n' "$NAME" "$DUR" "$LAST_LINK"

  # Post-build crate timing summary.  Strip ANSI escapes from log first
  # (cargo's `Compiling` lines come with bold-green colour codes that
  # break naive greps).  `--timings` writes a fresh HTML per build under
  # target/cargo-timings/ — newest by mtime is the one we just produced.
  STRIP='s/\x1B\[[0-9;]*[a-zA-Z]//g'
  if sed "$STRIP" "$LOG" | grep -q "Compiling "; then
    TIMING_HTML=$(ls -1t target/cargo-timings/cargo-timing-*.html 2>/dev/null | head -1)
    if [[ -n $TIMING_HTML ]]; then
      printf '   📊 per-crate timings: %s\n' "$(realpath "$TIMING_HTML")"
    fi
    # The LAST five `Compiling` lines before `Finished` — these are the
    # crates that gated the build (parallelism stragglers).
    LAST5=$(sed "$STRIP" "$LOG" | grep -E "^\s*Compiling " | tail -5 | sed 's/^\s*//')
    if [[ -n $LAST5 ]]; then
      printf '   ⏱  last 5 crates compiled (build tail):\n'
      printf '%s\n' "$LAST5" | sed 's/^/      /'
    fi
  fi
  exit 0
else
  DUR=$((SECONDS - START))
  printf '\033[1;31m❌ %-12s FAILED\033[0m  %ds  →  see %s\n' "$NAME" "$DUR" "$LAST_LINK"
  exit 1
fi
