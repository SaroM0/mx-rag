.PHONY: help install dev-install clean test test-cov lint format run docker-build docker-run docker-stop setup-env aws-deploy

help:
	@echo "Available commands:"
	@echo "  make help              Show this help message"
	@echo "  make install          Install production dependencies"
	@echo "  make dev-install      Install development dependencies"
	@echo "  make clean            Clean up Python cache files and virtual environment"
	@echo "  make test             Run tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make lint             Run linting checks"
	@echo "  make format           Format code using black"
	@echo "  make run              Run the FastAPI application locally"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run application in Docker"
	@echo "  make docker-stop      Stop Docker containers"
	@echo "  make setup-env        Create .env file from example"
	@echo "  make aws-deploy       Deploy to AWS using CDK"

install:
	pip install -r requirements.txt

dev-install:
	pip install pytest pytest-asyncio pytest-cov httpx black flake8

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	rm -rf venv/

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=term-missing

lint:
	flake8 src

format:
	black src

run:
	uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t mx-rag .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

setup-env:
	@if [ ! -f .env ]; then \
		if [ -f .env.example ]; then \
			cp .env.example .env; \
			echo ".env file created from .env.example"; \
			echo "Please edit .env file with your configuration"; \
		else \
			echo "Error: .env.example file not found"; \
			exit 1; \
		fi \
	else \
		echo ".env file already exists"; \
	fi

aws-deploy:
	cd infra && npm install && cdk deploy 