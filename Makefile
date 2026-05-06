.PHONY: build build-debug test lint fmt clippy check ci all deploy

# Production-grade build matrix used by CI-equivalent local check.
# `--locked` catches Cargo.lock drift (real incident 2026-05-02).
build:
	cargo build --release --locked

build-debug:
	cargo build --locked

test:
	cargo test --locked --all-targets --workspace

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --locked --all-targets --workspace -- -D warnings

lint: fmt clippy

# Full pre-push gate. Run before every PR.
ci: lint test build

all: ci

deploy:
	cd ~/deploy/krolik-server && \
	docker compose build --no-cache embed-server && \
	docker compose up -d --no-deps --force-recreate embed-server
