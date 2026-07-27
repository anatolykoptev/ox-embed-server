SHELL := /bin/bash
.PHONY: build build-debug test test-doc lint fmt clippy deny ci all logs logs-clean help

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

build:
	@$(CI_STAGE) build cargo build --release --locked --timings

build-debug:
	@$(CI_STAGE) build-debug cargo build --locked --timings

lint: fmt clippy

# Full pre-push gate. Run before every PR.
ci: lint deny test build
	@echo
	@printf '\033[1;32m═══════════════════════════════════════════\033[0m\n'
	@printf '\033[1;32m  ✅ CI gate green — ready to push\033[0m\n'
	@printf '\033[1;32m═══════════════════════════════════════════\033[0m\n'
	@printf 'Logs: %s/last-*.log\n' "$(LOGDIR)"

all: ci

# Show the most recent log for each stage. After a `make ci` run,
# operators can grep across all phases with `make logs | less`.
logs:
	@for stage in fmt clippy deny test-doc test build build-debug; do \
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
	@echo "  make test-doc     — cargo test --doc --locked                          (~5s warm)"
	@echo "  make test         — test-doc + cargo nextest run --locked --all-targets (~30s warm)"
	@echo "  make build        — cargo build --release --locked                     (~90s warm)"
	@echo "  make build-debug  — cargo build --locked"
	@echo "  make ci           — lint + deny + test + build  (full pre-push gate)   (~2 min warm)"
	@echo
	@echo "  make logs         — tail last log per stage from $(LOGDIR)"
	@echo "  make logs-clean   — wipe $(LOGDIR)"
