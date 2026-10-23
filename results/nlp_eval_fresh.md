# Text -> netlist: handwritten evaluation set (independent of synthetic training data) [handwritten_fresh.jsonl]

- Sentences: 20
- Token accuracy (gold tokenization): 1.000; macro F1 1.000
- Exact circuit match: 0.900
- Component P / R / F1: 0.981 / 0.963 / 0.972
- Valid netlists: 1.000
- Error types: {'components/values': 2}

| tag | P | R | F1 |
|---|---|---|---|
| SRC | 1.00 | 1.00 | 1.00 |
| RES | 1.00 | 1.00 | 1.00 |
| CAP | 1.00 | 1.00 | 1.00 |
| IND | 1.00 | 1.00 | 1.00 |
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
| 1 | a 9 volt battery with a 330R resistor and an LED in series | DC 9; S(resistor:330,diode:-) | DC 9; S(resistor:-,diode:-) | components/values |
| 6 | a pair of 10k resistors in parallel | -; P(resistor:10k,resistor:10k) | -; resistor:10k | components/values |
