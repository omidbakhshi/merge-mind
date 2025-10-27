# Contributing to Merge Mind

Thank you for your interest in contributing to Merge Mind! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/merge-mind.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt && pip install -e .[dev]`
6. Run tests: `pytest tests/`

## 🛠️ Development Setup

### Prerequisites
- Python 3.11+
- Git
- Docker (optional, for full testing)

### Local Development
```bash
# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Run linting
flake8 src tests
mypy src

# Start development server
uvicorn src.main:app --reload
```

## 📝 Code Style

We follow these coding standards:

- **Python**: PEP 8 with some modifications (max line length: 127)
- **Imports**: Grouped by standard library, third-party, local
- **Docstrings**: Google-style docstrings for all public functions
- **Type hints**: Required for all new code
- **Naming**: snake_case for functions/variables, PascalCase for classes

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
pip install pre-commit
pre-commit install
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_circuit_breaker.py

# Run performance tests
pytest tests/test_performance.py -v
```

### Writing Tests
- Use `pytest` framework
- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test names
- Mock external dependencies (OpenAI, GitLab APIs)

## 📚 Documentation

### Code Documentation
- All public functions/classes need docstrings
- Include type hints
- Document parameters, return values, and exceptions

### API Documentation
- FastAPI automatically generates OpenAPI docs
- Available at `/docs` when running the server
- Keep endpoint documentation up to date

## 🔄 Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality

4. **Run the full test suite**:
   ```bash
   pytest --cov=src
   flake8 src tests
   mypy src
   ```

5. **Update documentation** if needed

6. **Commit your changes**:
   ```bash
   git commit -m "feat: Add your feature description"
   ```

7. **Push to your fork** and **create a pull request**

### Commit Message Format

We follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

## 🐛 Reporting Issues

When reporting bugs, please include:

- **Description**: Clear description of the issue
- **Steps to reproduce**: Step-by-step instructions
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: Python version, OS, etc.
- **Logs**: Relevant log output

## 💡 Feature Requests

We welcome feature requests! Please:

- Check if the feature already exists or is planned
- Describe the use case clearly
- Explain why it would be valuable
- Consider implementation complexity

## 📄 License

By contributing to Merge Mind, you agree that your contributions will be licensed under the MIT License.

## 🤝 Code of Conduct

Please be respectful and constructive in all interactions. We aim to create a welcoming environment for all contributors.

## 📞 Getting Help

- **Documentation**: Check the README.md first
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions

Thank you for contributing to Merge Mind! 🎉