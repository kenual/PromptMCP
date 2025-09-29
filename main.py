"""
MCP Server Template
"""

import os
import glob
import inspect
import re
from typing import Any, Dict, List, Optional

import yaml

from mcp.server.fastmcp import FastMCP
from pydantic import Field


mcp = FastMCP("Prompt Server", stateless_http=True)


@mcp.tool(
    title="Echo Tool",
    description="Echo the input text",
)
def echo(text: str = Field(description="The text to echo")) -> str:
    return text



# ====== Dynamic recipe discovery and prompt registration ======

def _slugify(value: str) -> str:
    s = value.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "prompt"


def _coerce_type(t: Optional[str]):
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


def _build_and_register_prompt_from_recipe(
    title: str,
    description: str,
    instructions: str,
    template: str,
    parameters: List[Dict[str, Any]],
    source_name: str,
) -> None:
    # Runtime function that renders the prompt with simple {{var}} substitutions
    def _render(**kwargs):
        text_parts = []
        if instructions and instructions.strip():
            text_parts.append(instructions.strip())
        text_parts.append(template)
        text = "\n\n".join(text_parts)

        # very simple templating: replace {{key}} with provided value
        for k, v in kwargs.items():
            text = text.replace("{{" + k + "}}", "" if v is None else str(v))
        return text

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

    _render.__signature__ = inspect.Signature(
        parameters=sig_params, return_annotation=str)
    _render.__annotations__ = annotations
    _render.__name__ = f"prompt_{_slugify(title or source_name)}"
    _render.__doc__ = description or f"Prompt imported from recipe: {source_name}"

    # Register with MCP
    mcp.prompt(title or source_name)(_render)
    print(
        f"[PromptMCP] Registered prompt '{title or source_name}' from recipe '{source_name}'")


def _build_and_register_tool_from_recipe(
    title: str,
    description: str,
    instructions: str,
    template: str,
    parameters: List[Dict[str, Any]],
    source_name: str,
) -> None:
    # Runtime function that renders the output with simple {{var}} substitutions
    def _execute(**kwargs):
        text_parts = []
        if instructions and instructions.strip():
            text_parts.append(instructions.strip())
        text_parts.append(template)
        text = "\n\n".join(text_parts)

        # very simple templating: replace {{key}} with provided value
        for k, v in kwargs.items():
            text = text.replace("{{" + k + "}}", "" if v is None else str(v))
        return text

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

    _execute.__signature__ = inspect.Signature(
        parameters=sig_params, return_annotation=str)
    _execute.__annotations__ = annotations
    _execute.__name__ = f"tool_{_slugify(title or source_name)}"
    _execute.__doc__ = description or f"Tool imported from recipe: {source_name}"

    # Register with MCP
    mcp.tool(title=title or source_name, description=description)(_execute)
    print(
        f"[PromptMCP] Registered tool '{title or source_name}' from recipe '{source_name}'")

def _register_recipe_file(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[PromptMCP] Failed to load YAML '{path}': {e}")
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
        print(f"[PromptMCP] Skipping '{path}': recipe.prompt is missing/empty")
        return

    _build_and_register_prompt_from_recipe(
        title=title,
        description=description,
        instructions=instructions,
        template=template,
        parameters=params,
        source_name=os.path.basename(path),
    )

    _build_and_register_tool_from_recipe(
        title=title,
        description=description,
        instructions=instructions,
        template=template,
        parameters=params,
        source_name=os.path.basename(path),
    )


def load_prompts_from_recipes(dir_path: str = "recipes") -> None:
    if not os.path.isdir(dir_path):
        print(
            f"[PromptMCP] Recipes directory not found: {dir_path} (skipping)")
        return

    pattern_yaml = os.path.join(dir_path, "*.yaml")
    pattern_yml = os.path.join(dir_path, "*.yml")
    files = sorted(glob.glob(pattern_yaml) + glob.glob(pattern_yml))

    if not files:
        print(f"[PromptMCP] No recipe files found under {dir_path}")
        return

    for fp in files:
        _register_recipe_file(fp)


if __name__ == "__main__":
    # Discover and register recipe-based prompts before starting the server
    load_prompts_from_recipes("recipes")

    mcp.run(transport="streamable-http")
    # mcp.run(transport="stdio")
