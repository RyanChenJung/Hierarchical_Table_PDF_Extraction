# TableSight

Multi-model hierarchical table extraction with a side-by-side Streamlit dashboard.

Three pipelines, one interface:

| Model | Approach | Checkpoint |
|---|---|---|
| **Florence-2** | DETR-based vision-language model; LoRA fine-tuned on FinTabNet. Falls back to `<OCR_WITH_REGION>` + grid clustering when no adapter is loaded. | `microsoft/Florence-2-base` + LoRA |
| **UniTable** | 3-pass encoder-decoder: structure → bbox → per-cell content. Content model fine-tuned with LoRA. | poloclub/unitable + merged content LoRA |
| **SPARTAN** | OCR-first heuristic: EasyOCR word detection → 1D KMeans column clustering → header inference → HTML assembly. Reads tuned params from `models/spartan/*.json`. | none (parametric only) |

## Layout

```
TableSight/
├── README.md
├── requirements.txt
├── dashboard/
│   ├── app.py                            # Streamlit app — single entry point
│   ├── config.yaml                       # Checkpoint paths + device choice
│   └── utils/
│       ├── model_loader.py               # Loads + caches the 3 runners
│       ├── teds.py                       # TEDS scorer + failure classifier
│       └── visualization.py              # Plotly chart helpers
├── models/
│   ├── inference.py                      # Unified `TableExtractor` interface
│   ├── florence_runner.py
│   ├── unitable_runner.py
│   ├── spartan/
│   │   ├── pipeline.py                   # `SpartanRunner` (EasyOCR + cluster)
│   │   ├── prepare_data.py               # Build local FinTabNet splits from HF
│   │   ├── preprocessing_params.json     # Tuned params (consumed at load time)
│   │   ├── grid_params.json
│   │   └── spartan_benchmark.csv         # Test-split scores (169 images)
│   └── unitable_handoff/
│       ├── INTEGRATION_GUIDE.md          # Original handoff notes
│       ├── load_model_helper.py          # Provided loader
│       └── unitable/                     # Cloned poloclub/unitable repo
├── notebooks/
│   └── spartan_hierarchical_extractor.py # Train SPARTAN params + benchmark
```

## Running the dashboard

### Local install

```bash
pip install -r TableSight/requirements.txt
brew install tesseract                       # macOS — Linux: apt install tesseract-ocr
```

### Quick start (localhost only)

```bash
streamlit run TableSight/dashboard/app.py
```

Opens at **http://localhost:8501** with a browser auto-popup. Stop with `Ctrl+C`.

### LAN-accessible (share with anyone on your Wi-Fi)

```bash
streamlit run TableSight/dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
```

Bound to `0.0.0.0`, so it's reachable from other devices at
`http://<your-LAN-IP>:8501`. Find your IP with
`ipconfig getifaddr en0` (macOS) or `hostname -I | awk '{print $1}'` (Linux).

### Run as a persistent background service

```bash
nohup streamlit run TableSight/dashboard/app.py \
    --server.port 8501 --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false \
    > /tmp/tablesight.log 2>&1 &
echo $! > /tmp/tablesight.pid
```

Tail logs:

```bash
tail -f /tmp/tablesight.log
```

Stop:

```bash
kill $(cat /tmp/tablesight.pid)
# Or, if the PID file is gone:
kill $(lsof -ti :8501)
```

### Using it

Drop an image or pick one from the local FinTabNet gallery; click *Run* to
score the three models side-by-side. Tabs across the top:

- **Compare** — single-image inference + TEDS scoring per model
- **Benchmark** — aggregated stats from the saved evaluation run
- **Batch Eval** — run all loaded models over a CSV slice
- **Failure Analysis** — TEDS-below-threshold cases with diff + category pies

## Single-call API

```python
from TableSight.models.inference import TableExtractor
from PIL import Image

ext = TableExtractor("spartan",
                      checkpoint_path="TableSight/models/spartan")
out = ext.predict(Image.open("table.png"))
print(out["html"], out["time_s"], out["metadata"])
```

Available model names: `"florence"`, `"unitable"`, `"spartan"`.

## Training the SPARTAN params

The SPARTAN runner reads `preprocessing_params.json` + `grid_params.json` at
init. To retune them on a refreshed FinTabNet sample:

```bash
python TableSight/notebooks/spartan_hierarchical_extractor.py --train
```

The script auto-detects Colab vs local, bootstraps FinTabNet from HuggingFace
if no local cache exists, and writes the tuned JSON files into
`models/spartan/` — pickup takes effect on next dashboard reload.

## Checkpoints not in this repo

The Florence-2 LoRA adapter and UniTable bundle are 1–2 GB each and live
outside this repo. Update `dashboard/config.yaml` to point at wherever
they're mounted on the host:

```yaml
checkpoints:
  florence: "/path/to/florence2_phase1_general"   # PEFT adapter directory
  unitable: "/path/to/ryan_handoff"               # contains models/ + vocab/
  spartan:  "TableSight/models/spartan"           # in-repo (just JSON params)
```

The dashboard shows ✓/⚠ per model based on whether the checkpoint resolved.
