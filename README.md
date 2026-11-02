# cutting_manga_v1

**Find safe cut points in long-strip manhwa — with a self-trained image classifier.**

Long-strip webtoon chapters arrive as one tall image and must be split back into pages without
cutting through artwork or text. This repo trains and runs a small classifier that scores every
candidate slice line as **safe** (a clean gutter) or **unsafe** (content), then picks the best
split points.

> ภาพมังงะแนวยาวต้องหั่นกลับเป็นแผ่นโดยไม่ตัดโดนเนื้อหา — โปรเจคนี้เทรน classifier เองเพื่อให้คะแนน
> แต่ละจุดว่าตัดได้ (safe) หรือห้ามตัด (unsafe)

## How it works

1. **Window scoring** — a 512 px window slides down the strip (stride 16) and the model outputs a
   safety probability for each position.
2. **Smoothing + peak picking** — scores are smoothed, then `scipy.signal.find_peaks` selects the
   safest local maxima with a minimum distance between cuts.
3. **Split** — the image is cropped at the chosen `y` positions.

## Model

| | |
| --- | --- |
| Architecture | MobileNetV3-Small (`cnn_v5.pth`) |
| Fallback | EfficientNetB0 (older weights) |
| Input | 512 × 512 window |
| Classes | `1_safe` / `0_unsafe` |
| Throughput | ~1.5 s per image |

## Dataset

Built from scratch — see [`DATASET_GUIDE.md`](DATASET_GUIDE.md):

- **Ground-truth junctions** — bottom half of page *N* joined with top half of page *N+1* (a real
  cut) labelled `safe`
- **Random content sampling** — arbitrary slices through the artwork labelled `unsafe`
- **Hard-case mining** — false positives / false negatives found in real runs are re-labelled and
  fine-tuned with a low learning rate

Target ratio `unsafe : safe` ≈ 1.2–1.5.

## Usage

```bash
pip install -r requirements.txt

# GUI
run_app.bat          # or: python main.py

# Batch (merge -> predict -> split)
python auto_cut.py --input D:\dataMH --output D:\dataMH_out --workers 4 --quality 95
```

Input layout: `<INPUT>/<series>/<chapter_NNN>/<page>.jpg`
Output layout: `<OUTPUT>/<series>/<chapter_NNN>/<chapter_NNN>_<part>.jpg`

## Project layout

```
core/          ai_predictor.py (model + cut logic), image_ops.py, models.py
ui/            PySide desktop app
scripts/       dataset + maintenance scripts
tools/         helpers
docs/          model math / timeline notes
tests/         pytest contracts for the main window and split workspace
auto_cut.py    batch pipeline (multiprocessing workers)
main.py        GUI entry point
```

## Tests

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).
