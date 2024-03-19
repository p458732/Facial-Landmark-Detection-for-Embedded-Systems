#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import dataclasses
import re
import subprocess
from ast import Attribute, Call, Constant, Name, NodeVisitor, parse
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest


class CodeConventionError(Exception):
    pass


class CheckConventionsVisitor(NodeVisitor):
    def __init__(self, filename: str, *args, **kwargs) -> None:
        self.filename = filename
        super().__init__(*args, **kwargs)

    def visit_Call(self, node: Call) -> Any:
        if (
            isinstance(node.func, Name)
            and node.func.id in ("getattr", "setattr")
            and len(node.args) >= 2
            and isinstance(node.args[0], Name)
            and "opts" in node.args[0].id
            and isinstance(node.args[1], Constant)
            and isinstance(node.args[1].value, str)
            and "-" in node.args[1].value
        ):
            raise CodeConventionError(
                f"{self.filename}:{node.lineno} Please replace '-' with '_' in {node.func.id}() invocations."
            )

        if (
            isinstance(node.func, Attribute)
            and node.func.attr == "add_argument"
            and len(node.args) >= 1
            and isinstance(node.args[0], Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("--")
            and "_" in node.args[0].value
        ):
            raise CodeConventionError(
                f"{self.filename}:{node.lineno} Please replace '_' with '-' in add_argument() invocations."
            )

        return self.generic_visit(node)  # visit children


@pytest.mark.parametrize(
    "line,error",
    [
        ("getattr(opts, 'x_y')", None),
        ("setattr(opts, 'x_y', None)", None),
        ("parser.add_argument('--x-y')", None),
        ("group.add_argument('--x-y')", None),
        (
            "getattr(opts, 'x-y')",
            "Please replace '-' with '_' in getattr() invocations.",
        ),
        (
            "setattr(opts, 'x-y')",
            "Please replace '-' with '_' in setattr() invocations.",
        ),
        (
            "parser.add_argument('--x_y')",
            "Please replace '_' with '-' in add_argument() invocations.",
        ),
        (
            "group.add_argument('--x_y')",
            "Please replace '_' with '-' in add_argument() invocations.",
        ),
    ],
)
def test_validator(line: str, error: str) -> None:
    ast = parse(
        f"""
import argparse
parser = argparse.ArgumentParser()
group = parser.add_argument_group()
opts = parser.parse_args([])
{line}
"""
    )
    validator = CheckConventionsVisitor(filename="dummy-file")

    if error is None:
        validator.visit(ast)
        return

    with pytest.raises(CodeConventionError) as excinfo:
        validator.visit(ast)

    assert error in str(
        excinfo.value
    ), f"Expected {line} to raise CodeConventionError({error}) but it didn't."


def get_python_filenames() -> List[str]:
    return (
        subprocess.check_output(
            # `git ls-files` returns list of file names tracked by git
            # `git ls-files --others --exclude-standard` returns list of files neither tracked nor ignored by git
            "(git ls-files; git ls-files --others --exclude-standard) | grep '.py$'",
            shell=True,
            universal_newlines=True,
        )
        .strip()
        .splitlines(keepends=False)
    )


def test_python_filenames():
    filenames = get_python_filenames()
    assert len(filenames) > 10
    for filename in filenames:
        assert filename.endswith(".py"), f"filename={filename} should end with .py"


@dataclasses.dataclass
class ForbiddenRegex:
    regex: str
    handler: Callable[[str], str]
    exclude_dirs: List[str] = dataclasses.field(default_factory=list)


@pytest.mark.parametrize("filename", get_python_filenames())
def test_validate_python_files(filename: str) -> None:
    if __file__.endswith(filename):
        return  # errors in the current file are intentional

    source = Path(filename).read_text()

    forbidden_regexes = [
        ForbiddenRegex(
            regex=r"collate[-_\w]+eval|eval[-_\w]+collate|dataset[-_\w]+eval|eval[-_\w]+dataset",
            handler=lambda match: f"To follow the standard train/val/test dataset splitting terminology, please rename "
            f"'{match}' to either '{match.replace('eval', 'test')}' "
            f"or '{match.replace('eval', 'val')}'.",
        ),
        ForbiddenRegex(
            regex=r"from \.|import \.",
            handler=lambda match: f"Please use absolute imports instead of relative ones."
            f" For example: 'import .models' --> 'import cvnets.models'. Problematic code: '{match}'.",
        ),
        ForbiddenRegex(
            regex=r"class (ConvLayer|SeparableConv|TransposeConvLayer|ASPPConv|ASPPSeparableConv)\b",
            handler=lambda match: f"{match} is deprecated. Please remove it and use {match}2d.",
        ),
        ForbiddenRegex(
            regex=r"setattr\(opts.*model\.segmentation\.n_classes",
            handler=lambda match: f"'{match},...)' is deprecated. Please set it in config.",
            exclude_dirs=["tests/"],
        ),
    ]
    forbidden_regexes = {f"regex_{i}": item for i, item in enumerate(forbidden_regexes)}

    unified_regex = re.compile(
        "|".join(
            f"(?P<{name}>{item.regex})" for name, item in forbidden_regexes.items()
        )
    )

    if not filename.endswith("test_conventions.py"):
        match = unified_regex.search(source)
        if match:
            for name, match in match.groupdict().items():
                if not match:
                    continue
                for exclude_dir in forbidden_regexes[name].exclude_dirs:
                    if filename.startswith(exclude_dir):
                        break
                else:
                    message = forbidden_regexes[name].handler(match)
                    raise CodeConventionError(f"{message} File: {filename}.")

        ast = parse(source)
        CheckConventionsVisitor(filename=filename).visit(ast)


def get_regular_filenames() -> List[str]:
    # All regular text files
    return (
        subprocess.check_output(
            # lists all non-empty regular (no symlinks) text files:
            "git grep --cached -Il ''",
            shell=True,
            universal_newlines=True,
        )
        .strip()
        .splitlines(keepends=False)
    )


def test_regular_filenames():
    filenames = get_regular_filenames()
    assert len(filenames) > 10


@pytest.mark.parametrize("filename", get_regular_filenames())
def test_no_newline_before_EOF(filename: str) -> None:
    source = Path(filename).read_text()
    if source and source[-1] != "\n":
        raise CodeConventionError(
            f"Please add a new line at the end of regular files. File: {filename}"
        )
