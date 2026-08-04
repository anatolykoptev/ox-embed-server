SHELL := /bin/bash
.PHONY: build build-debug test test-doc lint fmt clippy deny secrets vulns audit mutants mutants-diff ci all logs logs-clean help

# Production-grade build matrix used by CI-equivalent local check.
# `--locked` catches Cargo.lock drift (real incident 2026-05-02).
#
# Every target runs through `scripts/ci-stage.sh` which:
#   * prints `▶ <name>` live (so the operator sees activity, not silence)
#   * tees stdout+stderr into a per-stage log under /tmp/embed-server-ci
#   * stops the chain on first failure (`set -o pipefail` inside)
#   * stamps elapsed seconds at the end
# This mirrors the deployment system's deploy pattern (logfile per phase + status notify).

CI_STAGE := scripts/ci-stage.sh
LOGDIR ?= /tmp/embed-server-ci

fmt:
	@$(CI_STAGE) fmt cargo fmt --all -- --check

clippy:
	@$(CI_STAGE) clippy cargo clippy --locked --all-targets --workspace -- -D warnings

# Doc-tests apply to library crates. embed-server is binary-only so this
# target is a no-op but kept for when a lib target is added.
test-doc:
	@echo "▶  test-doc   (no library target — skipped)"

test: test-doc
	@$(CI_STAGE) test cargo nextest run --locked --all-targets --workspace

deny:
	@$(CI_STAGE) deny cargo deny check

# ── security gates (mirror .github/workflows/preflight.yml) ─────────────────
# Each asserts its tool is installed rather than skipping: a gate that passes
# because the binary is absent is worse than no gate, because it reports green.

secrets:
	@command -v gitleaks >/dev/null || { echo "gitleaks not installed — https://github.com/gitleaks/gitleaks/releases (CI pins 8.30.1)"; exit 1; }
	@$(CI_STAGE) secrets gitleaks dir . --config .gitleaks.toml --no-banner --redact

vulns:
	@command -v osv-scanner >/dev/null || { echo "osv-scanner not installed — https://github.com/google/osv-scanner/releases (CI pins 2.2.4)"; exit 1; }
	@$(CI_STAGE) vulns osv-scanner scan source --lockfile=Cargo.lock --config=.osv-scanner.toml

audit: deny secrets vulns

# ── test-quality gate ──────────────────────────────────────────────────────
# Scope, test tool and timeouts all come from .cargo/mutants.toml — the same
# file CI reads, so a local run and the CI run cannot disagree.
#
# `mutants-diff` is the fast one and the one worth running before pushing: it
# mutates only the lines your branch changed, which is exactly what preflight
# gates on. `mutants` is the full scope — minutes to hours; that is the
# nightly lane's job, not a pre-push habit.

mutants-diff:
	@command -v cargo-mutants >/dev/null || { echo "cargo-mutants not installed — cargo install cargo-mutants@25.1.0"; exit 1; }
	@git diff origin/main...HEAD -- 'src/**/*.rs' > /tmp/embed-mutants.diff
	@if [ ! -s /tmp/embed-mutants.diff ]; then echo "▶  mutants-diff  (no src/ changes vs origin/main — skipped)"; else \
		$(CI_STAGE) mutants-diff cargo mutants --in-diff /tmp/embed-mutants.diff --no-shuffle -j 2; fi

mutants:
	@command -v cargo-mutants >/dev/null || { echo "cargo-mutants not installed — cargo install cargo-mutants@25.1.0"; exit 1; }
	@$(CI_STAGE) mutants cargo mutants --no-shuffle -j 2

build:
	@$(CI_STAGE) build cargo build --release --locked --timings

build-debug:
	@$(CI_STAGE) build-debug cargo build --locked --timings

lint: fmt clippy

# Full pre-push gate. Run before every PR.
#
# `mutants-diff` is deliberately NOT in this chain: it needs a pushed base to
# diff against and costs minutes, so it belongs to the CI lane and to a
# conscious local `make mutants-diff` before pushing — not to every `make ci`.
ci: lint audit test build
	@echo
	@printf '\033[1;32m═══════════════════════════════════════════\033[0m\n'
	@printf '\033[1;32m  ✅ CI gate green — ready to push\033[0m\n'
	@printf '\033[1;32m═══════════════════════════════════════════\033[0m\n'
	@printf 'Logs: %s/last-*.log\n' "$(LOGDIR)"

all: ci

# Show the most recent log for each stage. After a `make ci` run,
# operators can grep across all phases with `make logs | less`.
logs:
	@for stage in fmt clippy deny secrets vulns test-doc test build build-debug mutants mutants-diff; do \
		log="$(LOGDIR)/last-$$stage.log"; \
		if [[ -e $$log ]]; then \
			printf '\n\033[1;36m═══ %s ═══ %s\033[0m\n' "$$stage" "$$log"; \
			tail -n 40 "$$log"; \
		fi; \
	done

logs-clean:
	rm -rf $(LOGDIR)
	@echo "Cleared $(LOGDIR)"

help:
	@echo "embed-server local CI targets"
	@echo
	@echo "  make fmt          — cargo fmt --check                                  (~1s)"
	@echo "  make clippy       — cargo clippy + -D warnings                         (~5s warm)"
	@echo "  make lint         — fmt + clippy"
	@echo "  make deny         — cargo deny check (licenses, advisories, sources)   (~5s)"
	@echo "  make secrets      — gitleaks secrets scan                              (~3s)"
	@echo "  make vulns        — osv-scanner over Cargo.lock                        (~10s)"
	@echo "  make audit        — deny + secrets + vulns"
	@echo "  make test-doc     — cargo test --doc --locked                          (~5s warm)"
	@echo "  make test         — test-doc + cargo nextest run --locked --all-targets (~30s warm)"
	@echo "  make build        — cargo build --release --locked                     (~90s warm)"
	@echo "  make build-debug  — cargo build --locked"
	@echo "  make ci           — lint + audit + test + build  (full pre-push gate)  (~2 min warm)"
	@echo
	@echo "  make mutants-diff — cargo-mutants on the lines this branch changed     (minutes)"
	@echo "                      run this before pushing; it is what preflight gates on"
	@echo "  make mutants      — cargo-mutants over the full scope in .cargo/mutants.toml"
	@echo "                      (hours — this is the nightly lane's job)"
	@echo
	@echo "  make logs         — tail last log per stage from $(LOGDIR)"
	@echo "  make logs-clean   — wipe $(LOGDIR)"
