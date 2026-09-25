import time

from .utils import FunctionSpec, OutputType
from ai_scientist import claude_code


def query(
    system_message: str | None,
    user_message: str | list | None,
    func_spec: FunctionSpec | dict | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    # temperature / max_tokens are not configurable through Claude Code
    model = model_kwargs["model"]

    # Claude needs a user message: if we only have a system msg, use it as the user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    output_schema = None
    if func_spec is not None:
        # Function calling is emulated with structured output against the function's schema
        if isinstance(func_spec, FunctionSpec):
            schema, description = func_spec.json_schema, func_spec.description
        else:
            schema, description = func_spec["parameters"], func_spec.get("description")
        output_schema = {**schema, "description": description} if description else schema

    t0 = time.time()
    text, structured, usage = claude_code.query(
        [{"role": "user", "content": user_message}],
        model,
        system_message,
        output_schema=output_schema,
    )
    req_time = time.time() - t0

    output = structured if func_spec is not None else text
    in_tokens = (
        usage.get("input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
    )
    return output, req_time, in_tokens, usage.get("output_tokens", 0), {}
