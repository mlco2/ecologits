
.PHONY: install
install:
	uv sync --all-extras --all-groups

.PHONY: test
test:
	uv run pytest

.PHONY: test-record
test-record:
	uv run pytest --record-mode=once

.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: docs
docs:
	uv run --only-group docs zensical build -f mkdocs.yml --strict && uv run --only-group docs python scripts/check_docs_links.py site

.PHONY: serve-docs
serve-docs:
	uv run --only-group docs zensical serve -f mkdocs.yml
