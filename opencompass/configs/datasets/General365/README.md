# General365

The configuration downloads the public 720-example test split from
`meituan-longcat/General365_Public` at revision
`e6f063d0a74c51baa101204533823dd54b0c577b`. It presents each stored question
verbatim and evaluates the complete response set with the released hybrid
protocol: text and choice rows use a JSON-constrained judge, while all other
answer types use Math-Verify. The primary `average_accuracy` metric is on the
official 0-to-1 scale.

The checked-in `judge_cfg.path` defaults to `gpt-4.1-mini`, matching the
released executable scorer. Operators may override the path, key, and API base
through the normal OpenCompass configuration mechanism. Reproducing the
paper's reported conditions requires overriding the judge path with GPT-4.1;
the paper judge and the released scorer default are deliberately distinct.
The configured judge must support chat completions and JSON-object response
formatting.

## Reproducible scorer environment

General365's mathematical parser and verifier are version-sensitive. For
score reproduction and parity validation, use the benchmark-scoped lock copied
from the released scorer rather than OpenCompass's shared optional dependency
ranges:

```bash
python -m pip install \
  -r opencompass/configs/datasets/General365/requirements.lock
```

Run OpenCompass and the General365 alignment commands with that same Python
interpreter. The alignment adapter verifies every locked distribution before
loading data or scoring, and fails with an installation command if its
environment differs. The lock is deliberately benchmark-scoped because its
exact parser stack conflicts with dependency versions used by some unrelated
OpenCompass benchmarks.

## Upstream scorer provenance and license

The prompt and grading behavior is adapted from `grading.py` and `utils.py` in
[meituan-longcat/General365 at commit 6263ee4](https://github.com/meituan-longcat/General365/tree/6263ee4b995b205aa07f4c3ca5cbeb8e0e5acb43).

MIT License

Copyright (c) 2026 Meituan

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
