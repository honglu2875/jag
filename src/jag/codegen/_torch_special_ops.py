# Copyright (c) 2023 Honglu Fan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from jag.type import GraphNode

_kwargs_value_post_process = {}
_args_kwargs_post_process = {}
_special_ops = {}
_in_place_ops = set()


def register_value_process(name: str, key: str):
    def decorator(func):
        _kwargs_value_post_process.setdefault(name, {}).update({key: func})
        return func

    return decorator


def register_args_kwargs_post_process(name: str):
    def decorator(func):
        _args_kwargs_post_process[name] = func
        return func

    return decorator


def register_special_op(name: str, in_place=False):
    def decorator(func):
        _special_ops[name] = func
        if in_place:
            _in_place_ops.add(name)
        return func

    return decorator


@register_special_op("at")
def _process_at(node: GraphNode, operands_repr) -> str:
    assert len(node.operands) == 1, f"Expected 1 operand, got {len(node.operands)}."
    idx = node.kwargs["idx"]
    idx_repr = []
    for i in idx:
        if isinstance(i, int):
            idx_repr.append(str(i))
        elif isinstance(i, slice):
            if i.step is None or i.step == 1:
                idx_repr.append(f"{i.start or ''}:{i.stop or ''}")
            else:
                idx_repr.append(f"{i.start or ''}:{i.stop or ''}:{i.step}")
        else:
            raise ValueError(f"Unsupported index type: {type(i)}.")

    return f"{operands_repr[0]}[{', '.join(idx_repr)}]"


@register_special_op("replace", in_place=True)
def _process_replace(node: GraphNode, operands_repr) -> str:
    assert len(node.operands) == 2, f"Expected 2 operands, got {len(node.operands)}."
    idx = node.kwargs["idx"]
    idx_repr = []
    for i in idx:
        if isinstance(i, int):
            idx_repr.append(str(i))
        elif isinstance(i, slice):
            if i.step is None or i.step == 1:
                idx_repr.append(f"{i.start or ''}:{i.stop or ''}")
            else:
                idx_repr.append(f"{i.start or ''}:{i.stop or ''}:{i.step}")
        else:
            raise ValueError(f"Unsupported index type: {type(i)}.")

    return f"{operands_repr[0]}[{', '.join(idx_repr)}] = {operands_repr[1]}"


@register_special_op("unsqueeze")
def _process_unsqueeze(node: GraphNode, operands_repr) -> str:
    assert len(node.operands) == 1, f"Expected 1 operand, got {len(node.operands)}."
    idx = node.kwargs["axis"]
    if isinstance(idx, int):
        if 0 <= idx < 5:
            # unsqueeze a single index >=0 but not too far away (<5)
            return f"{operands_repr[0]}[{':, ' * idx + 'None, ...'}]"
        elif -3 <= idx < 0:
            # unsqueeze a single index <0 but not too far away (>=-3)
            return f"{operands_repr[0]}[{'..., None' + ', :' * (- idx + 1)}]"
        elif idx < 0:
            return f"torch.unsqueeze({operands_repr[0]}, axis={idx})"

        idx = (idx,)

    assert all(
        i >= 0 for i in idx
    ), f"Expect all indices to be >= 0 if axis is a tuple inside unsqueeze."
    assert len(set(idx)) == len(
        idx
    ), f"Expect all indices to be distinct if axis is a tuple inside unsqueeze."

    max_ids = max(idx) + 1
    idx = set(idx)

    index_str = []
    for i in range(max_ids - 1):
        index_str.append("None" if i in idx else ":")
    index_str.append("None")
    return f"{operands_repr[0]}[{', '.join(index_str)}]"


@register_value_process("reshape", "newshape")
def _reshape_newshape(value):
    if isinstance(value, int):
        return (value,)
    return value


@register_args_kwargs_post_process("where")
def _where_args_kwargs(args, kwargs):
    assert len(args) in (2, 3), f"Expected 2 or 3 arguments, got {len(args)}."
    if len(args) == 2:
        print(args, kwargs)
        return (kwargs["condition"], *args), {
            k: v for k, v in kwargs.items() if k != "condition"
        }
    else:
        return (args[2], args[0], args[1]), kwargs


@register_args_kwargs_post_process("transpose")
def _transpose_default_args(args, kwargs):
    if "axes" not in kwargs:
        kwargs["axes"] = range(args[0].ndim)[::-1]

    return args, kwargs
