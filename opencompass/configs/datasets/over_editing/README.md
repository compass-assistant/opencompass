# Over-Editing

Install OpenCompass with the evaluator dependencies copied from upstream
revision `701ad34febd436729251826d5cbb16fca5e278da`:

```bash
python -m pip install -e ".[over_editing]"
```

The upstream image also installs native build libraries. On Debian-like
systems, prepare those and the isolation executable with:

```bash
sudo apt-get install bubblewrap build-essential libgdal-dev libfreetype6-dev \
    libpng-dev pkg-config python3-dev
```

The English NLTK stopword corpus required by the selected tasks is packaged
from the pinned NLTK data snapshot documented in
`opencompass/datasets/over_editing/nltk_data/PROVENANCE.md`. The evaluator
mounts that data read-only inside bubblewrap; it does not rely on a user-home
download or download corpus data while scoring.

Evaluation fails before scoring if bubblewrap, the packaged stopword corpus,
or an import used by the selected programs and BigCodeBench tests is
unavailable inside the isolated runtime.
