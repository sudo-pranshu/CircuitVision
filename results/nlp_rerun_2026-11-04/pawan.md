# Text -> netlist: handwritten evaluation set (independent of synthetic training data) [handwritten_pawan.jsonl]

- Sentences: 20
- Token accuracy (gold tokenization): 0.994; macro F1 0.990
- Exact circuit match: 0.750
- Component P / R / F1: 0.971 / 1.000 / 0.986
- Valid netlists: 1.000
- Error types: {'tagging': 2, 'structure': 1, 'components/values': 2}

| tag | P | R | F1 |
|---|---|---|---|
| SRC | 1.00 | 1.00 | 1.00 |
| RES | 1.00 | 1.00 | 1.00 |
| CAP | 1.00 | 1.00 | 1.00 |
| IND | 1.00 | 1.00 | 1.00 |
| DIO | 1.00 | 1.00 | 1.00 |
| VAL | 1.00 | 1.00 | 1.00 |
| UNIT | 1.00 | 1.00 | 1.00 |
| SER | 1.00 | 0.94 | 0.97 |
| PAR | 1.00 | 0.92 | 0.96 |
| SEP | 0.93 | 1.00 | 0.97 |
| O | 0.99 | 1.00 | 1.00 |

## Failures

| # | sentence | expected | got | type |
|---|---|---|---|---|
| 6 | Two 220 ohm resistors are connected in series, and another 1k resistor is connected across both of them. | -; P(S(resistor:220,resistor:220),resistor:1k) | -; S(resistor:220,resistor:220,resistor:1k) | tagging |
| 10 | A 12 volt battery powers an LED through a 1k resistor. | DC 12; S(resistor:1k,diode:-) | DC 12; S(diode:-,resistor:1k) | tagging |
| 12 | Two 100 ohm resistors in parallel are connected in series with three 470 ohm resistors. | -; S(P(resistor:100,resistor:100),resistor:470,resistor:470,resistor:470) | -; S(resistor:100,resistor:100,resistor:470,resistor:470,resistor:470) | structure |
| 15 | A 9V battery is connected to an LED and a 330 ohms resistor in series, with a diode connected in parallel across the LED. | DC 9; S(P(diode:-,diode:-),resistor:330) | DC 9; S(diode:-,resistor:330,P(diode:-,diode:-)) | components/values |
| 19 | Two 4.7k resistors are connected in series with a 10uF capacitor connected parallel across the second resistor. | -; S(resistor:4.7k,P(capacitor:10u,resistor:4.7k)) | -; S(resistor:4.7k,resistor:4.7k,P(capacitor:10u,resistor:-)) | components/values |
