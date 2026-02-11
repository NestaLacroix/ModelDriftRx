.PHONY: help install lint format type-check test test-fast test-cov \
        run-api run-dashboard run-mlflow demo clean docker-up docker-down

# Default — show help
help:
	@echo ""
	@echo "  Model Drift Watch"
	@echo "  ────────────────────────────────────────"
	@echo ""
	@echo "  Setup:"
	@echo "    make install          Install all dependencies (dev mode)"
	@echo "    make clean            Remove caches and generated files"
	@echo ""
	@echo "  Quality:"
	@echo "    make lint             Run ruff linter"
	@echo "    make format           Auto-format code with ruff"
	@echo "    make type-check       Run mypy type checker"
	@echo "    make check            Run ALL checks (lint + format + types)"
	@echo ""
	@echo "  Testing:"
	@echo "    make test             Run full test suite"
	@echo "    make test-fast        Run tests, skip slow ones"
	@echo "    make test-cov         Run tests with coverage report"
	@echo ""
	@echo "  Run:"
	@echo "    make run-api          Start FastAPI server (hot reload)"
	@echo "    make run-dashboard    Start Streamlit dashboard"
	@echo "    make run-mlflow       Start MLflow tracking UI"
	@echo "    make demo             Run the full self-healing simulation"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-up        Start all services (API + dashboard + MLflow)"
	@echo "    make docker-down      Stop all services"
	@echo ""

# ================================ Setup =====================================

install:
	pip install -e ".[dev]"
	pre-commit install
	@echo "Installed. Run 'make test' to verify."

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage reports/
	@echo "Cleaned."

# ================================ Quality ===================================

lint:
	ruff check .

format:
	ruff format .
	@echo "Formatted."

type-check:
	mypy src/ api/

# Run ALL quality checks — this is what CI does
check: lint format type-check
	@echo "All checks passed."

# ================================ Testing ===================================

test:
	pytest -v

test-fast:
	pytest -v -m "not slow"

test-cov:
	pytest --cov --cov-report=term-missing --cov-report=html
	@echo "Coverage report: open htmlcov/index.html"

# ================================ Run =======================================

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py --server.port 8501

run-mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# ================================ Demo ======================================

demo:
	@echo "   Starting AutoPilot ML demo..."
	@echo "   This will:"
	@echo "   1. Train the example model"
	@echo "   2. Inject drift into the data"
	@echo "   3. Watch the system detect, diagnose, heal, and report"
	@echo ""
	python simulation/run_simulation.py

# ================================ Docker ====================================

docker-up:
	docker-compose up --build -d
	@echo "   Services running:"
	@echo "   API:        http://localhost:8000"
	@echo "   Dashboard:  http://localhost:8501"
	@echo "   MLflow:     http://localhost:5000"

docker-down:
	docker-compose down
	@echo "   Services stopped."