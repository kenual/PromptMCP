# AGENTS.md

Purpose
- This document guides agentic coding tools on how to build, run, lint, test, and contribute in this repository.
- It also defines code style and conventions to keep changes consistent and safe.

Project overview
- Language: Python (requires >= 3.13) from pyproject.toml:2-6
- Runtime tool: uv (see README.md:6-15)
- Entry point: main.py
- Recipes: .goose/recipes/*.yaml loaded at runtime
- No explicit dev tooling configured in pyproject.toml beyond runtime deps
- No Cursor rules or Copilot instructions detected

Repository layout
- main.py: server startup and recipe registration
- .goose/recipes/: YAML recipes the server loads
- README.md: quickstart commands
- pyproject.toml: project metadata and runtime dependencies
- .claude/skills/: documentation assets (non-runtime)
- No tests/ directory present yet

Key code references
- FastMCP server instance: main.py:19, main.py:302-310
- Recipe discovery: load_prompts_from_recipes in main.py:272-300
- Recipe registration: _register_recipe_file in main.py:215-269
- Callable construction:
  - _make_renderer in main.py:76-105
  - _compute_signature in main.py:108-155
  - _build_and_register_from_recipe in main.py:158-213
- Slug helper: _slugify in main.py:26-45
- Logging: logger = logging.getLogger("PromptMCP") in main.py:17

Environment and setup
- Python: >= 3.13
- Install in editable mode:
  - uv pip install -e .
- Run the server (streamable HTTP):
  - uv run main.py
- MCP Inspector (optional):
  - npx @modelcontextprotocol/inspector --transport http --server-url http://127.0.0.1:8000/mcp

Build, lint, typecheck, and test
- Build
  - No [build-system] configured; run from source via uv run.
  - If packaging is needed, add [build-system] and then use uv build.
- Lint
  - Not configured. If adding ruff:
    - uv pip install ruff
    - uv run ruff check .
    - Optional formatting: uv run ruff format
  - If adding black:
    - uv pip install black
    - uv run black .
- Typecheck
  - Not configured. If adding mypy:
    - uv pip install mypy
    - uv run mypy .
  - If adding pyright:
    - npm i -D pyright
    - npx pyright
- Tests
  - No tests/ directory currently exists.
  - Preferred: create tests/ using unittest (stdlib) or pytest.

Running tests (unittest)
- Discover all tests:
  - uv run python -m unittest discover -s tests -p "test_*.py"
- Run a module:
  - uv run python -m unittest tests.test_module
- Run a class:
  - uv run python -m unittest tests.test_module.TestClass
- Run a single test:
  - uv run python -m unittest tests.test_module.TestClass.test_method
- Pattern match (Python 3.13):
  - uv run python -m unittest -k "pattern"

Running tests (pytest)
- Install:
  - uv pip install pytest
- All tests:
  - uv run pytest -q
- Single file:
  - uv run pytest tests/test_file.py -q
- Single test:
  - uv run pytest tests/test_file.py::TestClass::test_method -q
- Pattern:
  - uv run pytest -k "pattern" -q

Runtime commands
- Start server (streamable HTTP):
  - uv run main.py
- Alternative transport (stdio) is present but commented: main.py:310
- Logging configured in __main__: main.py:303-307

Recipe conventions (.goose/recipes)
- Files discovered: *.yaml and *.yml: main.py:290-293
- Expected structure includes top-level key "recipe": main.py:236-246
- Recognized fields under recipe:
  - title/name/filename used for registration titles: main.py:239-244
  - description: main.py:241
  - instructions (optional preamble): main.py:242, main.py:76-105
  - prompt (template text): main.py:243, main.py:76-105
  - parameters (list): main.py:245-246
- Parameters schema (per _coerce_type and _compute_signature):
  - key: str (required)
  - input_type: one of string (default), number, integer/int, float, boolean/bool: main.py:64-73
  - requirement: "required" marks required; otherwise optional default None: main.py:137-145
  - description: used as Field metadata: main.py:138, main.py:142-144
- Registration behavior
  - Both a prompt and a tool are registered for each recipe: main.py:251-269
  - Names/titles are derived via slugging and kind: main.py:201-205

MCP callable construction
- Renderer closure performs simple {{var}} substitution: main.py:92-103
- Dynamic signature and annotations attached so FastMCP exposes parameters: main.py:196-203
- Prompt vs Tool registration uses appropriate decorators: main.py:207-212

Code style guidelines
- Imports
  - Order: stdlib, third-party, local
  - Absolute imports; avoid relative unless necessary
  - No wildcard imports
  - One module per line
- Formatting
  - Target line length <= 100 columns
  - Use triple-quoted docstrings for modules, classes, functions
  - Prefer f-strings for string interpolation
  - Minimize unnecessary blank lines
- Types
  - Add type hints to all public functions
  - Use built-in types (list, dict) or typing.List/Dict consistently within a file
  - Optional values as Optional[T] or T | None; be consistent within a file
  - For user-facing parameters, mirror type mapping used in _coerce_type: main.py:64-73
- Naming
  - Modules and functions: snake_case
  - Classes and Enums: PascalCase (e.g., MCPPrimitives): main.py:21-24
  - Constants: UPPER_CASE
  - Private helpers: leading underscore (e.g., _slugify, _compute_signature)
- Errors and logging
  - Do not use bare except; catch specific exceptions where possible
  - Use the module logger named "PromptMCP" via logging.getLogger("PromptMCP"): main.py:17
  - Include context (e.g., file path) in log messages as in main.py:207-212, 233-249, 286-299
  - Avoid print; rely on logging
  - Warn and continue on invalid recipe files (pattern in main.py)
- Function design
  - Separate pure helpers from I/O
  - Return early on invalid input
  - Document expected shapes of dicts and lists in docstrings
- YAML I/O
  - Use yaml.safe_load
  - Open files with encoding="utf-8"
  - Validate required fields; skip empty prompt templates with a warning: main.py:247-249
- Dependencies
  - Runtime deps in pyproject.toml [project.dependencies]
  - Keep dev tooling out of runtime deps; document commands here

Testing guidelines
- Place tests under tests/ with filenames starting with test_
- Mirror module structure; name tests after functions/classes
- For recipe-driven features, create temp YAML fixtures and point load_prompts_from_recipes to them
- Avoid real network calls; test function-level behavior where possible

Pre-PR checklist
- If using unittest:
  - uv run python -m unittest discover -s tests -p "test_*.py"
- If using pytest:
  - uv run pytest -q
- If configured:
  - uv run ruff check .
  - uv run ruff format --check
  - uv run mypy .
- Ensure README commands still work

Cursor/Copilot rules
- No Cursor rules found in .cursor/rules/ or .cursorrules
- No Copilot instructions found in .github/copilot-instructions.md
- If added later, mirror key constraints here

How to add dev tooling (optional)
- Ruff
  - uv pip install ruff
  - Add [tool.ruff] to pyproject.toml
  - Commands: uv run ruff check . and uv run ruff format
- Mypy
  - uv pip install mypy
  - Add [tool.mypy] to pyproject.toml
  - Command: uv run mypy .
- Pytest
  - uv pip install pytest
  - Tests under tests/
  - Single test: uv run pytest tests/test_file.py::TestClass::test_method -q

Notes for agents
- Follow existing helpers and patterns (_compute_signature, _make_renderer) instead of duplicating logic
- Maintain logging behavior and do not change external behavior without reason
- When adding commands or tooling, update this AGENTS.md and README.md accordingly
