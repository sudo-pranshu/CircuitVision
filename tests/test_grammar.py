from circuitvision.grammar import earley, tokenize, validate

GOOD = "* divider\nV1 1 0 DC 5\nR1 1 2 10k\nR2 2 0 1k\n.op\n.end\n"


def test_valid_netlist():
    assert validate(GOOD) == []


def test_ambiguous_token_node_vs_value():
    # "5" is a legal NODE and a legal VALUE; the parse must still succeed
    assert validate("* t\nV1 1 0 5\nR1 1 0 5\n.end\n") == []


def test_syntax_error_located():
    errs = validate(GOOD.replace("R1 1 2 10k", "R1 1 10k"))
    assert errs and "line 3" in errs[0]


def test_semantic_errors():
    assert "no ground node (0)" in validate("* t\nR1 1 2 1k\nR2 1 2 1k\n.end\n")
    assert any("floating" in e for e in validate(GOOD.replace("R2 2 0", "R2 3 0")))
    assert any("undefined model" in e for e in validate("* t\nV1 1 0 5\nD1 1 0 DMOD\n.end\n"))
    assert any("shorted" in e for e in validate("* t\nV1 1 0 5\nR1 1 0 1k\nR2 1 1 1k\n.end\n"))


def test_recognizer_rejects_missing_end():
    ok, _ = earley([t for t, _ in tokenize("* t\nR1 1 0 1k\n")])
    assert not ok
