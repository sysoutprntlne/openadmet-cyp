# Third-party notices

## Lizard Wizard Gizzard, CYP Challenge submission

- Source: https://github.com/doctawho42/CYPChallenge (commit `bf018b8311c5e7804674384aa5cc16b6c93db94b`)
- Used in: `src/representations/cyp_mech.py` (ported from `src/feats.py` and `src/pka.py`)
- What was taken: the functional-group SMARTS, the rule-based pKa class table and
  inductive-penalty weights, the dimorphite basic-site selection, and the definition
  and grouping of the 30 mechanistic columns.
- Changes: adapted to the `Representation` interface; invalid SMILES give NaN rows;
  dimorphite-dl 2.x API. Output verified identical to the original `mech_block` plus
  `dimorphite_charge` on a test panel.

The original license text follows.

---

MIT License

Copyright (c) 2026 the CYPChallenge contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
