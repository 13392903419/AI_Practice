"""
YOLOv8n-seg 训练脚本
用途：合并盲道 + 斑马线两个 Roboflow 数据集，训练 YOLOv8n-seg 模型
目标类别映射：{0: 'road_crossing', 1: 'blind_path'}

使用方法：
  python train_yolov8n_seg.py                    # 完整流程：下载+合并+训练
  python train_yolov8n_seg.py --skip-download     # 跳过下载，只合并+训练
  python train_yolov8n_seg.py --download-only      # 只下载，不训练
  python train_yolov8n_seg.py --epochs 30          # 覆盖训练轮数
  python train_yolov8n_seg.py --resume             # 从上次中断处继续训练
"""

import os
import sys
import shutil
import yaml
import argparse
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# 配置区
# ============================================================
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "datasets"
MERGED_DIR = DATASET_DIR / "merged_blindpath"

# Roboflow 项目信息
DATASETS = {
    "crosswalk": {
        "workspace": "new-workspace-picpt",
        "project": "crosswalk-seg-gl01a",
        "version": 1,
        "src_class_id": 0,   # 原始 class id
        "dst_class_id": 0,   # 映射后 class id → road_crossing
    },
    "blind_path": {
        "workspace": "new-workspace-picpt",
        "project": "aiglasses-9iroz",
        "version": 4,
        "src_class_id": 0,   # 原始 class id
        "dst_class_id": 1,   # 映射后 class id → blind_path
    },
}

# 最终类别定义（必须和 workflow_blindpath.py 里的 class_id 对应）
CLASS_NAMES = {0: "road_crossing", 1: "blind_path"}

# 训练超参数（针对 Tesla P4 7.5GB VRAM 调优）
TRAIN_CFG = {
    "model": "yolov8n-seg.pt",   # 预训练 nano 分割权重
    "imgsz": 640,                # 训练分辨率（推理时会用 320）
    "epochs": 150,
    "batch": 16,                 # P4 跑 nano + 640 够用
    "patience": 30,              # early stopping
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,                 # 最终学习率 = lr0 * lrf
    "weight_decay": 0.0005,
    "warmup_epochs": 5,
    "mosaic": 1.0,
    "mixup": 0.1,
    "degrees": 10.0,             # 旋转增强
    "translate": 0.2,
    "scale": 0.5,
    "flipud": 0.1,               # 上下翻转
    "fliplr": 0.5,               # 左右翻转
    "hsv_h": 0.015,
    "hsv_s": 0.5,
    "hsv_v": 0.3,
    "project": str(BASE_DIR / "runs" / "train"),
    "name": "blindpath_nano_seg",
    "exist_ok": True,
    "device": 0,
    "workers": 4,
    "half": True,                # FP16 训练
    "cos_lr": True,
}


# ============================================================
# 步骤1：从 Roboflow 下载数据集
# ============================================================
def download_datasets():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("[ERROR] 未找到 ROBOFLOW_API_KEY，请在 .env 中设置")
        sys.exit(1)

    import requests
    import zipfile
    import io

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for name, cfg in DATASETS.items():
        dst = DATASET_DIR / name
        if dst.exists() and any(dst.rglob("*.jpg")):
            print(f"\n[跳过] {name} 已存在: {dst}")
            downloaded[name] = dst
            continue

        print(f"\n{'='*50}")
        print(f"[下载] {name}: {cfg['workspace']}/{cfg['project']} v{cfg['version']}")
        print(f"{'='*50}")

        # 用 REST API 获取下载链接
        url = (
            f"https://api.roboflow.com/{cfg['workspace']}/{cfg['project']}"
            f"/{cfg['version']}/yolov8?api_key={api_key}"
        )
        print("  获取下载信息...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()

        # 检查 export 中的下载链接
        export_info = data.get("export", {})
        download_url = None
        if isinstance(export_info, dict):
            download_url = export_info.get("link")
        if not download_url:
            version_info = data.get("version", {})
            download_url = version_info.get("download")

        if not download_url:
            # REST API 无直接链接，回退到 SDK
            print("  REST API 无直接链接，回退到 Roboflow SDK...")
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(cfg["workspace"]).project(cfg["project"])
            project.version(cfg["version"]).download("yolov8", location=str(dst))
        else:
            print(f"  下载中: {download_url[:80]}...")
            resp = requests.get(download_url, stream=True, timeout=600)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            buf = io.BytesIO()
            downloaded_bytes = 0
            for chunk in resp.iter_content(chunk_size=8192):
                buf.write(chunk)
                downloaded_bytes += len(chunk)
                if total > 0:
                    pct = downloaded_bytes * 100 // total
                    mb_done = downloaded_bytes // 1024 // 1024
                    mb_total = total // 1024 // 1024
                    print(f"\r  进度: {mb_done}MB / {mb_total}MB ({pct}%)", end="", flush=True)
            print()

            # 解压
            dst.mkdir(parents=True, exist_ok=True)
            buf.seek(0)
            with zipfile.ZipFile(buf) as zf:
                zf.extractall(dst)
            print(f"  解压完成: {dst}")

        downloaded[name] = dst
        print(f"[OK] {name} 下载到: {downloaded[name]}")

    return downloaded


# ============================================================
# 步骤2：合并数据集 + 重映射类别
# ============================================================
def remap_label_file(src_path: Path, dst_path: Path, class_mapping: dict):
    """读取 YOLO 标签文件，重映射 class_id 后写入新位置"""
    lines = src_path.read_text(encoding="utf-8").strip().splitlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:  # 至少 class_id + 2 对坐标
            continue
        old_cls = int(parts[0])
        if old_cls in class_mapping:
            parts[0] = str(class_mapping[old_cls])
            new_lines.append(" ".join(parts))
    dst_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def merge_datasets():
    """合并多个数据集到统一目录，处理类别映射"""
    print(f"\n{'='*50}")
    print("[合并] 开始合并数据集")
    print(f"{'='*50}")

    # 清理旧的合并目录
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    splits = ["train", "valid", "test"]
    for split in splits:
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {s: {"images": 0, "labels": 0} for s in splits}

    for ds_name, cfg in DATASETS.items():
        ds_dir = DATASET_DIR / ds_name
        # 构建类别映射：原始 class_id → 目标 class_id
        class_mapping = {}

        # 先扫描 data.yaml 获取原始类名
        yaml_path = ds_dir / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                ds_yaml = yaml.safe_load(f)
            src_names = ds_yaml.get("names", {})
            if isinstance(src_names, list):
                src_names = {i: n for i, n in enumerate(src_names)}
            print(f"  [{ds_name}] 原始类别: {src_names}")

            # 对这个数据集，所有原始类都映射到目标 class_id
            for src_id in src_names:
                class_mapping[int(src_id)] = cfg["dst_class_id"]
        else:
            # fallback：直接用配置的映射
            class_mapping[cfg["src_class_id"]] = cfg["dst_class_id"]

        print(f"  [{ds_name}] 类别映射: {class_mapping}")

        for split in splits:
            img_src = ds_dir / split / "images"
            lbl_src = ds_dir / split / "labels"
            if not img_src.exists():
                print(f"  [{ds_name}] 跳过 {split}（目录不存在）")
                continue

            img_dst = MERGED_DIR / split / "images"
            lbl_dst = MERGED_DIR / split / "labels"

            for img_file in img_src.iterdir():
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue

                # 加前缀避免文件名冲突
                new_name = f"{ds_name}_{img_file.name}"
                shutil.copy2(img_file, img_dst / new_name)
                stats[split]["images"] += 1

                # 复制并重映射标签
                lbl_file = lbl_src / (img_file.stem + ".txt")
                if lbl_file.exists():
                    remap_label_file(
                        lbl_file, lbl_dst / f"{ds_name}_{lbl_file.name}",
                        class_mapping
                    )
                    stats[split]["labels"] += 1

    # 生成合并后的 data.yaml
    data_yaml = {
        "path": str(MERGED_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = MERGED_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[合并完成] 保存到: {MERGED_DIR}")
    for split, s in stats.items():
        print(f"  {split}: {s['images']} 张图片, {s['labels']} 个标签")
    print(f"  data.yaml: {yaml_path}")

    return yaml_path


# ============================================================
# 步骤3：训练
# ============================================================
def train(data_yaml: Path, resume: bool = False):
    print(f"\n{'='*50}")
    print("[训练] 开始 YOLOv8n-seg 训练")
    print(f"{'='*50}")

    from ultralytics import YOLO

    # 续训：从 last.pt 恢复
    if resume:
        last_pt = Path(TRAIN_CFG["project"]) / TRAIN_CFG["name"] / "weights" / "last.pt"
        if last_pt.exists():
            print(f"  [续训] 从 {last_pt} 恢复")
            model = YOLO(str(last_pt))
        else:
            print(f"  [续训] 未找到 {last_pt}，从头开始训练")
            model = YOLO(TRAIN_CFG["model"])
    else:
        model = YOLO(TRAIN_CFG["model"])

    train_args = {k: v for k, v in TRAIN_CFG.items() if k != "model"}
    train_args["data"] = str(data_yaml)
    if resume:
        train_args["resume"] = True

    print(f"  模型: {TRAIN_CFG['model']}")
    print(f"  数据: {data_yaml}")
    print(f"  Epochs: {train_args['epochs']}")
    print(f"  Batch: {train_args['batch']}")
    print(f"  ImgSz: {train_args['imgsz']}")

    results = model.train(**train_args)

    # 找到 best.pt
    best_pt = Path(train_args["project"]) / train_args["name"] / "weights" / "best.pt"
    if best_pt.exists():
        dst = BASE_DIR / "model" / "yolo-seg-nano.pt"
        shutil.copy2(best_pt, dst)
        print(f"\n[完成] best.pt 已复制到: {dst}")
        print(f"  要替换线上模型，修改 .env 中 BLIND_PATH_MODEL=model/yolo-seg-nano.pt")
    else:
        print(f"\n[警告] 未找到 best.pt: {best_pt}")

    return results


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="YOLOv8n-seg 盲道+斑马线 训练脚本")
    parser.add_argument("--skip-download", action="store_true", help="跳过数据集下载")
    parser.add_argument("--download-only", action="store_true", help="只下载不训练")
    parser.add_argument("--resume", action="store_true", help="从 last.pt 继续训练")
    parser.add_argument("--epochs", type=int, help="覆盖训练轮数")
    parser.add_argument("--batch", type=int, help="覆盖 batch size")
    parser.add_argument("--imgsz", type=int, help="覆盖训练图片尺寸")
    args = parser.parse_args()

    if args.epochs:
        TRAIN_CFG["epochs"] = args.epochs
    if args.batch:
        TRAIN_CFG["batch"] = args.batch
    if args.imgsz:
        TRAIN_CFG["imgsz"] = args.imgsz

    # 下载
    if not args.skip_download:
        download_datasets()
    else:
        print("[跳过] 数据集下载")

    if args.download_only:
        print("[完成] 仅下载模式，退出")
        return

    # 合并
    data_yaml = merge_datasets()

    # 训练
    train(data_yaml, resume=args.resume)


if __name__ == "__main__":
    main()
