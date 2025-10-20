"""
MCP Prompt Server
"""

import glob
import inspect
import logging
import os
import re
from enum import StrEnum
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field
import yaml

logger = logging.getLogger("PromptMCP")

mcp = FastMCP("Prompt Server", stateless_http=True)

class MCPPrimitives(StrEnum):
    prompt = "prompt"
    tool = "tool"


def _slugify(value: str) -> str:
    """
    Convert an arbitrary string into a lowercase, underscore-delimited slug.

    - Lowercases the input
    - Replaces runs of non-alphanumeric characters with a single underscore
    - Collapses repeated underscores and trims leading/trailing underscores
    Returns "prompt" if the normalized value would be empty.

    Args:
        value: Input string to normalize.

    Returns:
        A safe slug composed of [a-z0-9_] characters.
    """
    s = value.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "prompt"


def _coerce_type(t: Optional[str]):
    """
    Map a string type name to a Python type object.

    Supported values (case-insensitive):
    - "string" -> str (default)
    - "number" -> float
    - "integer"/"int" -> int
    - "float" -> float
    - "boolean"/"bool" -> bool

    Args:
        t: Type name from a recipe; None or unknown values default to "string".

    Returns:
        The corresponding Python type object.
    """
    t = (t or "string").strip().lower()
    return {
        "string": str,
        "number": float,
        "integer": int,
        "int": int,
        "float": float,
        "boolean": bool,
        "bool": bool,
    }.get(t, str)


def _make_renderer(instructions: str, template: str):
    """
    Create a simple renderer closure that interpolates variables into a template.

    The returned function accepts keyword arguments and performs naive template
    substitution by replacing occurrences of {{key}} with the provided value
    (or an empty string if the value is None). If non-empty, instructions are
    prepended above the template separated by a blank line.

    Args:
        instructions: Optional text placed before the template when rendering.
        template: The template text containing {{variable}} placeholders.

    Returns:
        A callable that renders the final string given keyword arguments.
    """
    # Returns a closure that renders text with simple {{var}} substitutions
    def _fn(**kwargs):
        text_parts = []
        if instructions and instructions.strip():
            text_parts.append(instructions.strip())
        text_parts.append(template)
        text = "\n\n".join(text_parts)

        # very simple templating: replace {{key}} with provided value
        for k, v in kwargs.items():
            text = text.replace("{{" + k + "}}", "" if v is None else str(v))
        return text

    return _fn


def _compute_signature(parameters: List[Dict[str, Any]]):
    """
    Compute inspect.Parameter objects and type annotations from recipe parameters.

    Each parameter dict may include:
    - key: Name of the argument.
    - input_type: One of the supported scalar types (see _coerce_type).
    - requirement: "required" to mark as required; anything else treated as optional.
    - description: Human-readable description used in the Field metadata.

    Required parameters receive Field(description=...), optional receive
    Field(default=None, description=...).

    Args:
        parameters: List of parameter dicts from a recipe.

    Returns:
        A tuple (sig_params, annotations) where:
        - sig_params: List[inspect.Parameter] suitable for building a Signature.
        - annotations: Dict[name, type] for __annotations__.
    """
    # Build a dynamic signature so MCP exposes parameters correctly
    sig_params = []
    annotations: Dict[str, Any] = {}
    for p in parameters or []:
        key = str(p.get("key", "")).strip()
        if not key:
            continue
        py_type = _coerce_type(p.get("input_type", "string"))
        requirement = str(p.get("requirement", "required")).strip().lower()
        desc = str(p.get("description", "")).strip()

        # Required if requirement == "required", else optional with default None
        if requirement == "required":
            default = Field(description=desc)
        else:
            default = Field(default=None, description=desc)

        annotations[key] = py_type
        sig_params.append(
            inspect.Parameter(
                name=key,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=py_type,
            )
        )
    return sig_params, annotations


def _build_and_register_from_recipe(
    kind: MCPPrimitives,
    title: str,
    description: str,
    instructions: str,
    template: str,
    parameters: List[Dict[str, Any]],
    source_name: str,
) -> None:
    """
    Build a callable from a recipe and register it with the MCP server as either
    a Prompt or a Tool.

    The function:
    - Creates a renderer closure that performs simple {{var}} substitutions.
    - Computes a dynamic function signature from the parameter definitions so
      FastMCP can expose typed parameters.
    - Attaches metadata (name, signature, annotations, docstring).
    - Registers the callable as a prompt or tool and prints a registration note.

    Args:
        kind: Whether to register as a prompt or a tool.
        title: Human-friendly title from the recipe (used for registration).
        description: Description used for the registered callable/docstring.
        instructions: Optional preamble text prepended when rendering.
        template: Template text containing {{placeholder}} tokens.
        parameters: List of parameter definitions from the recipe.
        source_name: Original filename of the recipe, used in messages/metadata.

    Returns:
        None. Registration is performed as a side effect.
    """
    # Create renderer closure
    fn = _make_renderer(instructions, template)

    # Build signature and annotations
    sig_params, annotations = _compute_signature(parameters)

    # Apply function metadata
    fn.__signature__ = inspect.Signature(
        parameters=sig_params, return_annotation=str)
    fn.__annotations__ = annotations

    prefix = kind.value
    capital = "Prompt" if kind is MCPPrimitives.prompt else "Tool"
    fn.__name__ = f"{prefix}_{_slugify(title or source_name)}"
    fn.__doc__ = description or f"{capital} imported from recipe: {source_name}"

    # Register with MCP
    if kind is MCPPrimitives.prompt:
        mcp.prompt(title or source_name)(fn)
        logger.info("Registered prompt '%s' from recipe '%s'", title or source_name, source_name)
    else:
        mcp.tool(title=title or source_name, description=description)(fn)
        logger.info("Registered tool '%s' from recipe '%s'", title or source_name, source_name)


def _register_recipe_file(path: str) -> None:
    """
    Load a single YAML recipe file and register both a prompt and a tool.

    The YAML is expected to contain a top-level "recipe" mapping with keys such as
    title/name/filename, description, instructions, prompt, and parameters.
    If the prompt/template text is missing or empty, the file is skipped.

    Args:
        path: Path to the YAML recipe file.

    Returns:
        None. Successful parsing results in registration side effects.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.exception("Failed to load YAML '%s'", path)
        return

    # Expected structure based on provided example
    recipe = data.get("recipe", {}) or {}

    title = recipe.get("title") or data.get("name") or data.get(
        "filename") or os.path.basename(path)
    description = recipe.get("description") or ""
    instructions = recipe.get("instructions") or ""
    template = recipe.get("prompt") or ""

    params = recipe.get("parameters") or []

    if not template:
        logger.warning("Skipping '%s': recipe.prompt is missing/empty", path)
        return

    _build_and_register_from_recipe(
        kind=MCPPrimitives.prompt,
        title=title,
        description=description,
        instructions=instructions,
        template=template,
        parameters=params,
        source_name=os.path.basename(path),
    )

    _build_and_register_from_recipe(
        kind=MCPPrimitives.tool,
        title=title,
        description=description,
        instructions=instructions,
        template=template,
        parameters=params,
        source_name=os.path.basename(path),
    )


def load_prompts_from_recipes(dir_path: str = ".goose/recipes") -> None:
    """
    Discover and register prompts/tools from recipe files in a directory.

    Scans the directory for *.yaml and *.yml files, loading each via
    _register_recipe_file. Prints diagnostic messages for missing directories
    and empty results.

    Args:
        dir_path: Directory containing recipe files. Defaults to ".goose/recipes".

    Returns:
        None.
    """
    if not os.path.isdir(dir_path):
        logger.warning("Recipes directory not found: %s (skipping)", dir_path)
        return

    pattern_yaml = os.path.join(dir_path, "*.yaml")
    pattern_yml = os.path.join(dir_path, "*.yml")
    files = sorted(glob.glob(pattern_yaml) + glob.glob(pattern_yml))

    if not files:
        logger.warning("No recipe files found under %s", dir_path)
        return

    for fp in files:
        _register_recipe_file(fp)


if __name__ == "__main__":
    # Discover and register recipe-based prompts before starting the server
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    load_prompts_from_recipes(".goose/recipes")

    mcp.run(transport="streamable-http")
    # mcp.run(transport="stdio")
