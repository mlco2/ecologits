
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
	uv run mkdocs build

.PHONY: serve-docs
serve-docs:
	uv run mkdocs serve
