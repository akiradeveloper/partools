default:
    @just --list

doc:
    cargo doc -p massively --no-deps
    python3 -m http.server --directory target/doc 3000

doc-down:
    fuser -k 3000/tcp 2>/dev/null || true

bench:
    cargo bench -p massively

performance:
    python3 scripts/render-performance.py --run

test-api:
    cargo doc -p massively --no-deps
    bash scripts/check-public-api.sh

test-oracle:
    cargo nextest run -p oracle

lint:
    cargo clippy --workspace --lib -- -D warnings -A clippy::type_complexity -A clippy::too_many_arguments -A clippy::unnecessary_cast -A clippy::manual_is_multiple_of

test: test-api
    cargo nextest run
    cargo test -p massively --doc

publish:
    cargo publish -p massively
