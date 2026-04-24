# guardrails/__init__.py
def check(question, answer, context):
    from guardrails.checker import check as _check
    return _check(question, answer, context)

__all__ = ["check"]