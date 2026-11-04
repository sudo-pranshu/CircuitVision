# Text -> netlist: handwritten evaluation set (independent of synthetic training data) [handwritten_eval.jsonl]

- Sentences: 30
- Token accuracy (gold tokenization): 0.995; macro F1 0.980
- Exact circuit match: 1.000
- Component P / R / F1: 1.000 / 1.000 / 1.000
- Valid netlists: 1.000
- Error types: none

| tag | P | R | F1 |
|---|---|---|---|
| SRC | 0.95 | 1.00 | 0.97 |
| RES | 1.00 | 0.98 | 0.99 |
| CAP | 0.92 | 1.00 | 0.96 |
| IND | 1.00 | 0.75 | 0.86 |
| DIO | 1.00 | 1.00 | 1.00 |
| VAL | 1.00 | 1.00 | 1.00 |
| UNIT | 1.00 | 1.00 | 1.00 |
| SER | 1.00 | 1.00 | 1.00 |
| PAR | 1.00 | 1.00 | 1.00 |
| SEP | 1.00 | 1.00 | 1.00 |
| O | 1.00 | 1.00 | 1.00 |

## Failures

| # | sentence | expected | got | type |
|---|---|---|---|---|
