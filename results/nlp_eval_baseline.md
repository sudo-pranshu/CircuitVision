# Text -> netlist: handwritten evaluation set (independent of synthetic training data)

- Sentences: 30
- Token accuracy (gold tokenization): 0.986; macro F1 0.975
- Exact circuit match: 0.800
- Component P / R / F1: 0.965 / 0.965 / 0.965
- Valid netlists: 1.000
- Error types: {'tokenization': 3, 'structure': 2, 'tagging': 1}

| tag | P | R | F1 |
|---|---|---|---|
| SRC | 0.86 | 1.00 | 0.93 |
| RES | 1.00 | 0.93 | 0.96 |
| CAP | 1.00 | 1.00 | 1.00 |
| IND | 1.00 | 0.75 | 0.86 |
| DIO | 1.00 | 1.00 | 1.00 |
| VAL | 1.00 | 0.97 | 0.99 |
| UNIT | 1.00 | 1.00 | 1.00 |
| SER | 1.00 | 1.00 | 1.00 |
| PAR | 1.00 | 1.00 | 1.00 |
| SEP | 1.00 | 1.00 | 1.00 |
| O | 0.98 | 1.00 | 0.99 |

## Failures

| # | sentence | expected | got | type |
|---|---|---|---|---|
| 9 | a 4k7 resistor in series with a 2R2 resistor driven by a 9 V cell | DC 9; S(resistor:4.7k,resistor:2.2) | DC 9; S(resistor:4k,resistor:2) | tokenization |
| 15 | a 3.3k resistor and a 6.8k resistor in parallel, in series with a 1k resistor | -; S(P(resistor:3.3k,resistor:6.8k),resistor:1k) | -; S(resistor:3.3k,resistor:6.8k,resistor:1k) | structure |
| 16 | a 1k resistor in series with a 2k resistor, all in parallel with a 3k resistor | -; P(S(resistor:1k,resistor:2k),resistor:3k) | -; S(resistor:1k,P(resistor:2k,resistor:3k)) | structure |
| 20 | a 560 ohm resistor followed by a red LED followed by a 1N4148 diode powered by a 12V battery | DC 12; S(resistor:560,diode:-,diode:-) | DC 12; S(resistor:560,diode:-,resistor:1n,diode:-) | tokenization |
| 21 | two 1k resistors in series across a 5V supply | DC 5; S(resistor:1k,resistor:1k) | DC 5; resistor:1k | tagging |
| 25 | (a 1k resistor in series with a 1k resistor) in parallel with a 500 ohm resistor | -; P(S(resistor:1k,resistor:1k),resistor:500) | -; S(resistor:1k,P(resistor:1k,resistor:500)) | tokenization |
