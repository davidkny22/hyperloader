"""Stable identities for callables and configured Python objects."""

from __future__ import annotations

import hashlib
import inspect
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from typing import Any


def callable_identity(value: Any) -> dict[str, object]:
    """Return a qualname and code-object hash without capturing closure values."""
    if value is None:
        return {"kind": "none"}
    if isinstance(value, type):
        return {"kind": "type", "qualname": _qualified_name(value)}
    if inspect.isbuiltin(value) or inspect.ismethoddescriptor(value):
        return {"kind": "builtin", "qualname": _qualified_name(value)}
    if isinstance(value, partial):
        return {
            "args": stable_value(value.args),
            "callable": callable_identity(value.func),
            "keywords": stable_value(value.keywords or {}),
            "kind": "partial",
        }
    target = value.__func__ if inspect.ismethod(value) else value
    code = getattr(target, "__code__", None)
    if code is not None:
        return {
            "code_sha256": _code_hash(code),
            "kind": "function",
            "qualname": _qualified_name(target),
        }
    call = getattr(type(value), "__call__", None)
    call_code = getattr(call, "__code__", None)
    return {
        "call_sha256": None if call_code is None else _code_hash(call_code),
        "config": _public_state(value),
        "kind": "callable-object" if callable(value) else "object",
        "qualname": _qualified_name(type(value)),
    }


def stable_value(value: Any, *, _depth: int = 0) -> Any:
    """Project configuration-like values into canonical JSON data."""
    if _depth > 12:
        return {"qualname": _qualified_name(type(value)), "truncated": True}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, Enum):
        return stable_value(value.value, _depth=_depth + 1)
    if isinstance(value, type):
        return {"type": _qualified_name(value)}
    if isinstance(value, (tuple, list)):
        return [stable_value(item, _depth=_depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [stable_value(item, _depth=_depth + 1) for item in value]
        return sorted(items, key=repr)
    if isinstance(value, dict):
        return {
            str(key): stable_value(item, _depth=_depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if callable(value):
        return callable_identity(value)
    return {
        "qualname": _qualified_name(type(value)),
        "state": _public_state(value, _depth=_depth + 1),
    }


def _public_state(value: Any, *, _depth: int = 0) -> dict[str, object]:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: stable_value(getattr(value, field.name), _depth=_depth + 1)
            for field in fields(value)
            if not field.name.startswith("_")
        }
    try:
        state = vars(value)
    except TypeError:
        return {}
    return {
        name: stable_value(item, _depth=_depth + 1)
        for name, item in sorted(state.items())
        if not name.startswith("_")
    }


def _qualified_name(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def _code_hash(code: types.CodeType) -> str:
    payload = repr(_code_payload(code)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _code_payload(code: types.CodeType) -> tuple[object, ...]:
    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        code.co_code.hex(),
        tuple(_constant(value) for value in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
    )


def _constant(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return _code_payload(value)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, tuple):
        return tuple(_constant(item) for item in value)
    if isinstance(value, frozenset):
        return tuple(sorted((_constant(item) for item in value), key=repr))
    return {"type": _qualified_name(type(value))}
