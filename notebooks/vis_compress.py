import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")

with app.setup:
    import os
    from pathlib import Path
    from PIL import Image
    import io
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt


@app.cell
def load_data():
    import random as _random
    _root = Path('/dataset/IMAGENET-HACKATON-2026')
    _train_dir = _root / 'train'
    _all_classes = sorted([d.name for d in _train_dir.iterdir() if d.is_dir()])

    _random.seed(42)
    _selected_classes = _random.sample(_all_classes, 25)
    image_paths = []
    class_names = []
    for _cls in _selected_classes:
        _imgs = sorted(Path(_train_dir / _cls).glob('*.JPEG'))
        if _imgs:
            image_paths.append(_imgs[0])
            class_names.append(_cls)

    print(f"Loaded {len(image_paths)} images from different classes")
    return (image_paths,)


@app.cell
def quality_slider():
    q = mo.ui.slider(5, 100, step=5, value=80, label='JPEG Quality')
    q
    return (q,)


@app.cell
def compress_fn(image_paths, q):
    def compress_jpeg(img_path, quality):
        img = Image.open(img_path)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality, optimize=True)
        sz = buf.tell()
        buf.seek(0)
        return Image.open(buf), sz

    print(f"JPEG Quality = {q.value}")
    print(f"Images: {len(image_paths)}")
    _total_orig = sum(os.path.getsize(p) for p in image_paths)
    _total_comp = sum(compress_jpeg(p, q.value)[1] for p in image_paths)
    print(f"Total: {_total_orig//1024} KB -> {_total_comp//1024} KB ({(1-_total_comp/_total_orig)*100:.1f}% reduction)")
    return (compress_jpeg,)


@app.cell
def comparison(compress_jpeg, image_paths):
    _quals = [5, 25, 50, 75, 95]
    # Average across all 25 images
    _rows = []
    for _q in _quals:
        _total_orig = 0
        _total_sz = 0
        for _img_path in image_paths:
            _, _sz = compress_jpeg(_img_path, _q)
            _total_orig += os.path.getsize(_img_path)
            _total_sz += _sz
        _rows.append({
            'Quality': _q,
            'Avg Size (KB)': round(_total_sz / len(image_paths) / 1024, 1),
            'Avg Reduction %': round((1 - _total_sz / _total_orig) * 100, 1)
        })

    mo.ui.table(_rows, label='Average Compression Across 25 Images')
    return


@app.cell
def visualization(compress_jpeg, image_paths, q):
    _fig, _axes = plt.subplots(5, 5, figsize=(15, 15))

    for _idx in range(25):
        _row = _idx // 5
        _col = _idx % 5
        _img_path = image_paths[_idx]
        _img, _sz = compress_jpeg(_img_path, q.value)
        _orig = os.path.getsize(_img_path)
    
        _axes[_row, _col].imshow(_img)
        _axes[_row, _col].set_title(f"{_orig//1024}K->{_sz//1024}K", fontsize=7)
        _axes[_row, _col].axis('off')

    plt.tight_layout()
    _fig.suptitle(f"25 Images at JPEG Quality = {q.value}", fontsize=14, y=1.01)
    _fig
    return


if __name__ == "__main__":
    app.run()
