"""
Bridge between alphapareto Expression format and AlphaAgent function_lib format.

Self-contained: NO external dependency on alphapareto package.

Parses LLM output in alphapareto style (e.g. "Std($close,10)")
and translates to function_lib style (e.g. "TS_STD($close, 10)")
so the existing eval() pipeline can execute it.
"""

import re

# =====================================================================
# Operator definitions (mirrors alphapareto config.py)
# =====================================================================

# category -> {alphapareto_name: (funclib_name, n_args)}
UNARY_OPS = {
    "Abs": ("ABS", 1),
    "Log": ("LOG", 1),
}

BINARY_OPS = {
    "Add": ("+", 2),       # infix in function_lib
    "Sub": ("-", 2),
    "Mul": ("*", 2),
    "Div": ("/", 2),
    "Greater": ("MAX", 2), # MAX(A, B) in function_lib
    "Less": ("MIN", 2),    # MIN(A, B) in function_lib
}

ROLLING_OPS = {
    "Ref":   ("DELAY", 2),
    "Mean":  ("TS_MEAN", 2),
    "Sum":   ("TS_SUM", 2),
    "Std":   ("TS_STD", 2),
    "Var":   ("TS_VAR", 2),
    "Max":   ("TS_MAX", 2),
    "Min":   ("TS_MIN", 2),
    "Med":   ("TS_MEDIAN", 2),
    "Mad":   ("TS_MAD", 2),
    "Delta": ("DELTA", 2),
    "WMA":   ("WMA", 2),
    "EMA":   ("EMA", 2),
}

PAIR_ROLLING_OPS = {
    "Cov":  ("TS_COVARIANCE", 3),
    "Corr": ("TS_CORR", 3),
}

ALL_OPS = {}
ALL_OPS.update(UNARY_OPS)
ALL_OPS.update(BINARY_OPS)
ALL_OPS.update(ROLLING_OPS)
ALL_OPS.update(PAIR_ROLLING_OPS)

VALID_FEATURES = {"open", "close", "high", "low", "volume", "vwap"}

# =====================================================================
# Tokenizer
# =====================================================================

_TOKEN_RE = re.compile(
    r"""
      (\$[a-zA-Z_]\w*)       # feature like $close
    | ([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?d?)  # number, optional 'd' suffix stripped
    | ([a-zA-Z_]\w*)          # identifier (operator name)
    | ([(),])                 # punctuation
    """,
    re.VERBOSE,
)


def _tokenize(expr_str):
    """Tokenize an alphapareto expression string."""
    tokens = []
    for m in _TOKEN_RE.finditer(expr_str):
        feature, number, ident, punct = m.groups()
        if feature:
            tokens.append(("FEATURE", feature))
        elif number is not None and number != "":
            # Strip optional 'd' suffix (e.g. "10d" -> "10")
            tokens.append(("NUMBER", number.rstrip("d")))
        elif ident:
            tokens.append(("IDENT", ident))
        elif punct:
            tokens.append(("PUNCT", punct))
    return tokens


# =====================================================================
# Recursive descent parser → AST
# =====================================================================

class ASTNode:
    pass

class FeatureNode(ASTNode):
    def __init__(self, name):
        self.name = name  # e.g. "close"
    def __repr__(self):
        return f"${self.name}"

class ConstantNode(ASTNode):
    def __init__(self, value):
        self.value = float(value)
    def __repr__(self):
        if self.value == int(self.value) and abs(self.value) < 1e10:
            return str(int(self.value))
        return str(self.value)

class OpNode(ASTNode):
    def __init__(self, op_name, args):
        self.op_name = op_name  # alphapareto name, e.g. "Std"
        self.args = args
    def __repr__(self):
        return f"{self.op_name}({','.join(repr(a) for a in self.args)})"


class ParseError(Exception):
    pass


class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type=None, expected_value=None):
        tok = self.peek()
        if tok is None:
            raise ParseError(f"Unexpected end of expression")
        if expected_type and tok[0] != expected_type:
            raise ParseError(f"Expected {expected_type}, got {tok}")
        if expected_value and tok[1] != expected_value:
            raise ParseError(f"Expected '{expected_value}', got '{tok[1]}'")
        self.pos += 1
        return tok

    def parse_expr(self):
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected end of expression")

        typ, val = tok

        if typ == "FEATURE":
            self.consume()
            name = val[1:]  # strip $
            if name not in VALID_FEATURES:
                raise ParseError(f"Unknown feature: {val}")
            return FeatureNode(name)

        elif typ == "NUMBER":
            self.consume()
            return ConstantNode(val)

        elif typ == "IDENT":
            if val not in ALL_OPS:
                raise ParseError(f"Unknown operator: {val}")
            self.consume()
            # expect '('
            self.consume("PUNCT", "(")
            args = []
            # parse arguments
            if self.peek() and self.peek()[1] != ")":
                args.append(self.parse_expr())
                while self.peek() and self.peek()[1] == ",":
                    self.consume("PUNCT", ",")
                    args.append(self.parse_expr())
            self.consume("PUNCT", ")")

            expected_nargs = ALL_OPS[val][1]
            if len(args) != expected_nargs:
                raise ParseError(
                    f"{val} expects {expected_nargs} args, got {len(args)}"
                )
            return OpNode(val, args)

        elif typ == "PUNCT" and val == "(":
            # Parenthesized sub-expression (shouldn't normally appear, but handle it)
            self.consume()
            node = self.parse_expr()
            self.consume("PUNCT", ")")
            return node

        else:
            raise ParseError(f"Unexpected token: {tok}")

    def parse(self):
        result = self.parse_expr()
        if self.pos < len(self.tokens):
            raise ParseError(
                f"Unexpected trailing tokens: {self.tokens[self.pos:]}"
            )
        return result


def _parse(expr_str):
    """Parse an alphapareto expression string into an AST."""
    tokens = _tokenize(expr_str)
    if not tokens:
        raise ParseError(f"Empty expression: {expr_str!r}")
    parser = _Parser(tokens)
    return parser.parse()


# =====================================================================
# AST → function_lib format string
# =====================================================================

def _ast_to_funclib(node):
    """Convert AST to function_lib compatible string."""
    if isinstance(node, FeatureNode):
        return f"${node.name}"

    if isinstance(node, ConstantNode):
        v = node.value
        if v == int(v) and abs(v) < 1e10:
            return str(int(v))
        return str(v)

    if isinstance(node, OpNode):
        op = node.op_name
        args_str = [_ast_to_funclib(a) for a in node.args]

        # Binary ops with infix notation
        if op == "Add":
            return f"({args_str[0]} + {args_str[1]})"
        elif op == "Sub":
            return f"({args_str[0]} - {args_str[1]})"
        elif op == "Mul":
            return f"({args_str[0]} * {args_str[1]})"
        elif op == "Div":
            return f"({args_str[0]} / {args_str[1]})"

        # All other ops: function call
        funclib_name = ALL_OPS[op][0]
        return f"{funclib_name}({', '.join(args_str)})"

    raise ValueError(f"Unknown AST node: {type(node)}")


# =====================================================================
# AST → canonical alphapareto format string
# =====================================================================

def _ast_to_alphapareto(node):
    """Convert AST to canonical alphapareto format string."""
    if isinstance(node, FeatureNode):
        return f"${node.name}"

    if isinstance(node, ConstantNode):
        v = node.value
        if v == int(v) and abs(v) < 1e10:
            return str(int(v))
        return str(v)

    if isinstance(node, OpNode):
        args_str = [_ast_to_alphapareto(a) for a in node.args]
        return f"{node.op_name}({','.join(args_str)})"

    raise ValueError(f"Unknown AST node: {type(node)}")


# =====================================================================
# Public API
# =====================================================================

def translate(expr_str):
    """
    alphapareto format string → function_lib format string.

    >>> translate("Std($close,10)")
    'TS_STD($close, 10)'
    >>> translate("Add(Abs($open),Mul($high,0.5))")
    '(ABS($open) + ($high * 0.5))'
    """
    ast = _parse(expr_str)
    return _ast_to_funclib(ast)


def normalize(expr_str):
    """
    Parse and re-serialize to canonical alphapareto format.

    >>> normalize("Std( $close , 10 )")
    'Std($close,10)'
    >>> normalize("Add($open, Sub($high, $low))")
    'Add($open,Sub($high,$low))'
    """
    ast = _parse(expr_str)
    return _ast_to_alphapareto(ast)


def validate(expr_str):
    """
    Validate an expression. Returns (True, canonical_str) or (False, error_msg).
    """
    try:
        ast = _parse(expr_str)
        return True, _ast_to_alphapareto(ast)
    except ParseError as e:
        return False, str(e)
