# Contributing

Thanks for your interest in contributing to this DSP tutorial! Here's how to get started.

## Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/<your-username>/signal_processing_tutorial.git
cd signal_processing_tutorial

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

Each module follows this convention:

```
XX-module-name/
├── README.md          # Theory with LaTeX equations and Mermaid diagrams
├── module_script.py   # Python implementation with examples
└── exercises.ipynb    # (optional) Interactive exercises
```

## Code Style

- Follow existing patterns in the codebase
- Use NumPy-style docstrings for all public functions
- Include type information in docstring `Args` sections
- Use `matplotlib` for all visualizations with consistent styling:
  - `plt.grid(True, alpha=0.3)` for grid lines
  - `plt.tight_layout()` before `plt.show()`
  - Labeled axes and descriptive titles
- Keep CPU-only code in Modules 1-10; GPU code belongs in Modules 11-13

## Submitting Changes

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and verify:
   ```bash
   # Check that all scripts compile
   python -m py_compile path/to/your_script.py

   # Run the script to verify output
   python path/to/your_script.py
   ```

3. **Commit and push:**
   ```bash
   git add <files>
   git commit -m "Add description of your changes"
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** against `main` with a clear description.

## Reporting Issues

- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) for bugs
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) for new ideas
- Include your Python version, OS, and relevant package versions

## What to Contribute

- Fix typos or errors in theory documentation
- Add new examples or exercises to existing modules
- Improve visualizations or code clarity
- Add new modules covering related DSP topics
- Performance improvements or alternative implementations
