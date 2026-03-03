# P3-SAM API — Deployment Guide

目標機器：**NVIDIA RTX Pro 6000 (Blackwell, sm_120)**，CUDA 12.8，nvcc 12.8

---

## 目錄

1. [前置需求](#1-前置需求)
2. [專案結構](#2-專案結構)
3. [Build Docker Image](#3-build-docker-image)
4. [啟動服務](#4-啟動服務)
5. [模型 Weights 說明](#5-模型-weights-說明)
6. [API 端點](#6-api-端點)
7. [常用維運指令](#7-常用維運指令)
8. [常見問題排查](#8-常見問題排查)
9. [Build 參數速查](#9-build-參數速查)

---

## 1. 前置需求

### 主機端

| 項目 | 需求 |
|------|------|
| OS | Ubuntu 20.04 / 22.04 |
| GPU | NVIDIA RTX Pro 6000（sm_120） |
| NVIDIA Driver | ≥ 570（支援 CUDA 12.8） |
| Docker | ≥ 24.0 |
| NVIDIA Container Toolkit | 已安裝並設為 default runtime |
| 磁碟空間（build） | ≥ 30 GB（含中間層） |
| 磁碟空間（weights） | ≥ 5 GB（P3-SAM + Sonata） |
| `/pegaai` 掛載 | 已掛載（與 ReconViaGen 共用 HuggingFace cache） |

### 確認 NVIDIA Container Toolkit 是否就緒

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

---

## 2. 專案結構

Build context 必須是 **`Hunyuan3D-Part/`** 上層目錄，因為 Dockerfile 需要同時 COPY `P3-SAM/` 和 `XPart/partgen/`（Sonata feature extractor）。

```
Hunyuan3D-Part/                ← Docker build context
├── .dockerignore
├── P3-SAM/
│   ├── Dockerfile             ← 主 Dockerfile
│   ├── docker-compose.yml
│   ├── docker-build.sh
│   ├── requirements-api.txt
│   ├── DEPLOYMENT.md
│   ├── demo/
│   │   ├── p3sam_api.py       ← API 入口（WORKDIR 在此）
│   │   └── auto_mask.py
│   ├── model.py               ← P3SAM 模型定義（含 HF_HOME 修正）
│   └── utils/
│       └── chamfer3D/         ← 自訂 CUDA extension（build 時編譯）
└── XPart/
    └── partgen/
        └── models/
            └── sonata/        ← Sonata feature extractor（本地 Python 套件）
```

---

## 3. Build Docker Image

### 方法 A：使用 helper script（推薦）

```bash
# 從 Hunyuan3D-Part/ 目錄執行
./P3-SAM/docker-build.sh
```

加速選項——只編 sm_120（僅針對這台機器，build 時間減少約 30%）：

```bash
./P3-SAM/docker-build.sh p3sam:latest 4 "12.0"
#                        ↑tag         ↑jobs ↑arch
```

### 方法 B：手動 docker build

```bash
cd Hunyuan3D-Part/

docker build \
    -t p3sam:latest \
    -f P3-SAM/Dockerfile \
    --build-arg TORCH_CUDA_ARCH_LIST="12.0" \
    --build-arg MAX_JOBS=4 \
    .
```

### Proxy 設定

Proxy 已**預設寫入** `http://proxy.intra:80`（Dockerfile ARG、docker-compose build args、docker-build.sh 三者一致），與 ReconViaGen / qwen-image 相同，無需額外設定。

若需換成其他 proxy：

```bash
# docker-build.sh 方式
http_proxy=http://other-proxy:3128 ./P3-SAM/docker-build.sh

# 手動 docker build 方式
--build-arg http_proxy="http://other-proxy:3128" \
--build-arg https_proxy="http://other-proxy:3128"
```

若完全不需要 proxy（直接連外網）：

```bash
http_proxy="" https_proxy="" ./P3-SAM/docker-build.sh
```

### Build 各步驟說明

| Step | 內容 | 預計時間 |
|------|------|----------|
| 1 | PyTorch 2.7.1 + cu128 wheel | 3–5 min |
| 2 | spconv-cu120 | 1 min |
| 3 | PyG packages (torch-scatter 等) | 1–2 min |
| 4 | flash-attn 2.8.3（source build） | **20–40 min** |
| 5 | Pure Python deps | 3–5 min |
| 6 | COPY source code | < 1 min |
| 7 | Build chamfer3D CUDA extension | 2–5 min |
| **Total** | | **約 35–60 min** |

---

## 4. 啟動服務

### 方法 A：docker compose（推薦）

```bash
# 從 Hunyuan3D-Part/ 目錄
docker compose -f P3-SAM/docker-compose.yml up -d
```

> `/pegaai/model_team/huggingface_cache` 需在主機上已存在（通常 ReconViaGen 已建立）。
> 若不存在：`mkdir -p /pegaai/model_team/huggingface_cache`

### 方法 B：docker run（一次性測試）

```bash
docker run -d \
    --gpus all \
    --name p3sam-api \
    -p 5001:5001 \
    --shm-size=8g \
    -e HF_HOME=/root/.cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface \
    -e SPCONV_ALGO=native \
    -v /pegaai/model_team/huggingface_cache:/root/.cache/huggingface \
    p3sam:latest
```

### 確認服務已啟動

```bash
curl http://localhost:5001/health
# 預期：{"status":"ok","model_available":true}
```

---

## 5. 模型 Weights 說明

服務採用**懶下載**策略：第一次收到請求時才從 HuggingFace 下載模型，下載後快取到 volume，之後重啟不需再下載。

兩個模型都遵循標準 HuggingFace 快取慣例，路徑統一由 `HF_HOME` env var 控制：

| 模型 | HuggingFace Repo | 容器內快取路徑 |
|------|------------------|----------------|
| P3-SAM weights | `tencent/Hunyuan3D-Part` | `$HF_HOME/hub/models--tencent--Hunyuan3D-Part/` |
| Sonata feature extractor | `facebook/sonata` | `$HF_HOME/sonata/` |

容器內 `HF_HOME` = `/root/.cache/huggingface`，經由 volume 掛載對應到主機的 `/pegaai/model_team/huggingface_cache`，與 ReconViaGen 共用同一份目錄。

### 預先下載（可選，避免首次請求等待）

若主機可直接連上 HuggingFace，可在容器啟動後手動觸發：

```bash
docker exec -it p3sam-api python - <<'EOF'
from huggingface_hub import hf_hub_download
import os, sys
sys.path.insert(0, '..')
from model import build_P3SAM   # 觸發 Sonata 的 download_root 計算

# 下載 P3-SAM weights
hf_hub_download(repo_id="tencent/Hunyuan3D-Part", filename="p3sam/p3sam.safetensors")
print("✅ P3-SAM weights downloaded.")

# 下載 Sonata weights
hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
sonata_cache = os.path.join(hf_home, 'sonata')
from models import sonata
sonata.load("sonata", repo_id="facebook/sonata", download_root=sonata_cache)
print("✅ Sonata weights downloaded.")
EOF
```

---

## 6. API 端點

服務啟動後，Swagger UI 可在 `http://<host-ip>:5001/docs` 存取。

### `GET /health`

確認服務是否正常運行。

```bash
curl http://localhost:5001/health
```

```json
{"status": "ok", "model_available": true}
```

---

### `POST /segment`

上傳 3D 模型，回傳分割結果。

**接受格式**：`.glb`、`.ply`、`.obj`

**Form 參數**：

| 欄位 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `file` | — | — | 上傳的 3D 模型檔案（必填） |
| `point_num` | `100000` | 1000–500000 | 點雲取樣數量，越大越精確但越慢 |
| `prompt_num` | `400` | 10–1000 | 分割 Prompt 數量 |
| `threshold` | `0.95` | 0.0–1.0 | 分割信心閾值 |
| `post_process` | `true` | — | 是否套用後處理 |
| `clean_mesh` | `true` | — | 推論前是否清理 Mesh |
| `seed` | `42` | — | 隨機種子，控制結果可重現性 |
| `prompt_bs` | `32` | 1–400 | Prompt 推理 batch size，越大越快但佔用更多 VRAM |

**curl 範例**：

```bash
curl -X POST http://localhost:5001/segment \
    -F "file=@your_model.glb" \
    -F "point_num=100000" \
    -F "threshold=0.95"
```

**回應範例**：

```json
{
    "segmented_glb": "/download/550e8400-e29b-41d4-a716-446655440000/segmented_output_parts.glb",
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "num_parts": 7
}
```

---

### `GET /download/{request_id}/{file_name}`

下載分割結果的 GLB 檔案。

```bash
curl -O http://localhost:5001/download/550e8400-e29b-41d4-a716-446655440000/segmented_output_parts.glb
```

---

### 錯誤代碼

| HTTP | `error_code` | 說明 | 處理建議 |
|------|--------------|------|----------|
| 422 | — | 不支援的檔案格式 | 改用 `.glb` / `.ply` / `.obj` |
| 503 | `GPU_OOM` | GPU VRAM 不足 | 等候 30 秒後重試；降低 `point_num` 或 `prompt_bs` |
| 503 | `MODEL_UNAVAILABLE` | 模型載入失敗 | 等候並重試；查看 logs |
| 507 | `DISK_FULL` | 磁碟空間不足 | 手動清理 `/tmp` 或輸出目錄 |
| 500 | `INFERENCE_ERROR` | 未知推論錯誤 | 查看 logs |

所有 503 回應包含 `Retry-After: 30` header。

---

## 7. 常用維運指令

### 查看 logs

```bash
docker compose -f P3-SAM/docker-compose.yml logs -f p3sam
# 或
docker logs -f p3sam-api
```

### 進入容器

```bash
docker compose exec p3sam bash
# 或
docker exec -it p3sam-api bash
```

### 確認 GPU 是否被容器使用

```bash
docker exec p3sam-api nvidia-smi
```

### 停止服務

```bash
docker compose -f P3-SAM/docker-compose.yml down
```

### 重新 build 並重啟

```bash
docker compose -f P3-SAM/docker-compose.yml up -d --build
```

### 查看容器 health status

```bash
docker inspect p3sam-api --format='{{.State.Health.Status}}'
```

### 手動清理容器內暫存輸出

分割結果存在容器內 `/tmp/` 下的 UUID 目錄，容器關閉時由 FastAPI shutdown hook 自動清除。如需手動清理：

```bash
docker exec p3sam-api find /tmp -maxdepth 1 -type d -name '*-*-*' -exec rm -rf {} +
```

---

## 8. 常見問題排查

### Build 失敗：`torch-scatter` 找不到 cu126 wheel

PyG 可能尚未發布對應 wheel，在 `Dockerfile` 的 STEP 3 改為從 source build：

```dockerfile
RUN pip install --no-cache-dir \
    git+https://github.com/rusty1s/pytorch_scatter.git
```

> 注意：source build 需額外約 10–15 分鐘。

---

### 啟動後 `/health` 回傳 `model_available: false`

表示 `AutoMask` import 失敗，通常是 CUDA extension 或 Python 路徑問題。

```bash
docker logs p3sam-api 2>&1 | head -50
```

常見原因：
- `chamfer3D` 編譯失敗 → 確認 build log 中 Step 6 是否有錯誤
- `spconv` 版本問題 → `docker exec p3sam-api python -c "import spconv; print(spconv.__version__)"`
- `torch_scatter` 未正確安裝 → `docker exec p3sam-api python -c "import torch_scatter"`

---

### 第一次請求非常慢（幾分鐘）

正常現象。第一次請求會觸發：

1. 從 HuggingFace 下載 P3-SAM weights（`tencent/Hunyuan3D-Part`，~1 GB）
2. 從 HuggingFace 下載 Sonata weights（`facebook/sonata`）

兩者都快取到 `/pegaai/model_team/huggingface_cache`，之後重啟不再下載。
每次請求仍會 load/unload 模型到 GPU（這是設計行為，確保 VRAM 在請求結束後釋放）。

---

### GPU OOM（503 GPU_OOM）

降低請求參數：

```bash
curl -X POST http://localhost:5001/segment \
    -F "file=@model.glb" \
    -F "point_num=50000" \
    -F "prompt_bs=16"
```

---

### `illegal instruction` 或 SIGILL 錯誤

某個 CUDA extension 的 kernel 不支援 sm_120。確認 build 時 `TORCH_CUDA_ARCH_LIST` 有包含 `12.0`：

```bash
docker inspect p3sam:latest --format='{{json .Config.Env}}' | tr ',' '\n' | grep TORCH
```

若缺少，重新 build：

```bash
./P3-SAM/docker-build.sh p3sam:latest 4 "8.0;8.6;8.9;9.0;10.0;12.0"
```

---

### HuggingFace 下載失敗（網路不通）

確認容器可連外：

```bash
docker exec p3sam-api curl -I https://huggingface.co
```

若在 proxy 環境，在 `docker-compose.yml` 的 `environment` 加上：

```yaml
environment:
  - HTTPS_PROXY=http://proxy.intra:80
  - HTTP_PROXY=http://proxy.intra:80
  - NO_PROXY=localhost,127.0.0.1
```

---

## 9. Build 參數速查

| ARG | 預設值 | 說明 |
|-----|--------|------|
| `TORCH_CUDA_ARCH_LIST` | `"8.0;8.6;8.9;9.0;10.0;12.0"` | 編譯 CUDA kernel 的目標架構。設為 `"12.0"` 可加速 build，僅針對 RTX Pro 6000 |
| `MAX_JOBS` | `4` | ninja 平行編譯 job 數。核心多的機器可調高（注意 RAM 用量） |
| `http_proxy` | `""` | 企業 proxy URL，留空表示不使用 |
| `https_proxy` | `""` | 同上 |
| `no_proxy` | `"localhost,127.0.0.1"` | 不走 proxy 的位址 |
