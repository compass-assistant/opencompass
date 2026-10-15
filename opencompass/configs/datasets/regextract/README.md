# RegExtract

The default configuration downloads the public IBM PFAS-regulations
repository at commit `842a78ca45a3d426a5b4613181268236b5c60593` from
<https://github.com/IBM/PFAS-regulations>. It loads the 20 paired
`golden_dataset_txts/document-NN.txt` and
`golden_dataset_json/document-NN.json` files and the published Joint
Extraction prompt from `prompts.py`.

The dataset configuration fixes the published output limit at 8192 tokens.
Pair it with a model configuration that sends `temperature=0`; temperature is
a model-adapter setting in OpenCompass and cannot be set by a dataset config.

Reference annotations remain lossless JSON text in the Arrow dataset because
the published nested values have heterogeneous types. The registered evaluator
decodes each reference before applying the published scoring rules.

Set `COMPASS_DATA_CACHE` to use a different OpenCompass data cache. An
already prepared checkout can be used by passing its directory as `path` in
the dataset configuration.
