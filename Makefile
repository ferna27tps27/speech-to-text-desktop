.PHONY: run test test-integration lint typecheck install install-dev clean

# Run the voice agent
run:
	python -m voice_agent

# Run the voice agent (backward-compatible entry point)
run-legacy:
	python gemini_agent.py

# Run unit tests (no API keys required)
test:
	python -m pytest tests/ -v --ignore=tests/integration

# Run integration tests (requires GEMINI_API_KEY in .env)
test-integration:
	python -m pytest tests/integration/ -v

# Run all tests
test-all:
	python -m pytest tests/ -v

# Lint with ruff
lint:
	python -m ruff check voice_agent/ tests/

# Type check with mypy
typecheck:
	python -m mypy voice_agent/

# Install runtime dependencies
install:
	pip install -r requirements.txt

# Install all dependencies (runtime + test + dev)
install-dev:
	pip install -e ".[dev]"

# Clean bytecode and caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
