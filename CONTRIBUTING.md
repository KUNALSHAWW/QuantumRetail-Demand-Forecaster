# Contributing to QuantumRetail Demand Forecaster

First off, thank you for considering contributing to QuantumRetail Demand Forecaster! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Message Guidelines](#commit-message-guidelines)

---

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

---

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (Python version, OS, etc.)
- **Screenshots** (if applicable)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear title**
- **Provide detailed description**
- **Explain why this would be useful**
- **Include examples** (if applicable)

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:
- `good-first-issue` - Good for newcomers
- `help-wanted` - Extra attention needed

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv/conda)

### Setup Steps

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/FreshRetailNet-50k-Forecast.git
cd FreshRetailNet-50k-Forecast

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black flake8 mypy
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Running the Application

```bash
cd streamlit-app
streamlit run app.py
```

---

## 🔄 Pull Request Process

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following coding standards

4. **Test your changes** thoroughly

5. **Commit** using conventional commit format:
   ```bash
   git commit -m "feat: add new forecasting model"
   ```

6. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** with:
   - Clear title and description
   - Reference to related issues
   - Screenshots (if UI changes)
   - Testing evidence

8. **Address review feedback** promptly

---

## 💻 Coding Standards

### Python Style Guide

- Follow **PEP 8** style guide
- Use **type hints** where applicable
- Maximum line length: **88 characters** (Black default)
- Use **descriptive variable names**

### Code Formatting

We use **Black** for code formatting:

```bash
# Format all Python files
black src/ streamlit-app/

# Check formatting
black --check src/ streamlit-app/
```

### Linting

We use **Flake8** for linting:

```bash
flake8 src/ streamlit-app/ --max-line-length=88
```

### Type Checking

We use **MyPy** for type checking:

```bash
mypy src/
```

### Documentation

- Add **docstrings** to all functions and classes
- Use **Google-style** docstrings:

```python
def time_series_split(df: pd.DataFrame, train_window: int):
    """
    Split DataFrame into train and validation sets.
    
    Args:
        df: Input DataFrame with temporal data
        train_window: Number of days for training
        
    Returns:
        Tuple of (train_df, validation_df)
        
    Raises:
        ValueError: If train_window exceeds available data
    """
    # Implementation
```

---

## 📝 Commit Message Guidelines

We follow **Conventional Commits** specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
feat(model): add LSTM forecasting model

Implement LSTM-based deep learning model for time series forecasting.
Includes hyperparameter tuning and performance comparison with existing models.

Closes #42

---

fix(app): resolve caching issue in Streamlit

Fixed model caching logic that caused incorrect predictions
when switching between stores.

---

docs(readme): update installation instructions

Added troubleshooting section for common installation issues.
```

---

## 🌳 Branch Naming

Use descriptive branch names:

- `feature/add-lstm-model`
- `fix/caching-bug`
- `docs/update-readme`
- `refactor/model-selection`

---

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files: `test_<module_name>.py`
- Use descriptive test function names:

```python
def test_latent_demand_recovery_with_stockout():
    """Test latent demand recovery calculates correctly during stockouts."""
    # Test implementation
```

### Test Coverage

- Aim for **>80% code coverage**
- Test edge cases and error conditions
- Include integration tests for pipelines

---

## 📊 Performance Considerations

- Profile code for bottlenecks
- Use **caching** for expensive operations
- Optimize DataFrame operations
- Document performance characteristics

---

## 🚀 Release Process

Releases follow **Semantic Versioning** (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

---

## 📧 Questions?

- Open an issue with label `question`
- Contact: [@KUNALSHAWW](https://github.com/KUNALSHAWW)

---

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

---

**Thank you for contributing to QuantumRetail Demand Forecaster!** 🎉

Every contribution, no matter how small, is valuable and appreciated.
