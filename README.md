# SlatForge Server

A minimal GPU‑accelerated TCP backend that converts single images into textured 3D meshes using TRELLIS.2‑4B.

## Features

- Image to 3D mesh generation
- Automatic image preprocessing (crop + resize to 512×512)
- Texture generation
- Decimation, remeshing, bounding‑box cleanup
- Configurable sampling parameters
- Automatic Hugging Face model caching
- Optimized VRAM usage

## Requirements

- Windows 10 or 11
- RAM: 32 GB recommended; 16 GB may work but startup will be very slow.
- Modern NVIDIA GPU with 8+ GB VRAM (RTX 3050 8GB or newer) | 12+ GB is recommended for medium/high voxel resolution
- ~32 GB disk space for program files and AI models

## Installation

Download the pre‑compiled standalone server from the Releases page and extract the archive to your desired location.

## Running

Execute start_server.bat. The AI models will be downloaded automatically (first run only – may take several minutes). Wait for the pipeline initialization; it will be done when "SlatForge Server is ready to receive jobs." appear on the console.
Listens on 127.0.0.1:50007. Logs to log.txt.

## API

The server accepts a single JSON object per connection, sent as UTF‑8. It responds with a JSON object.

### Common Request Format

```json
{
  "action": "<action_name>",
  ... other parameters ...
}
```

### Actions

#### `test_connection`
Checks if the server is alive.

**Request:**
```json
{ "action": "test_connection" }
```

**Response:**
```json
{ "status": "ok", "message": "Connection successful" }
```

#### `paste_image`
Gets an image from the system clipboard, preprocesses it (optional), and saves it to the `temp` folder.

**Parameters:**
- `preprocess` (bool, default `true`) – if `true`, runs preprocessing (crop + resize to 512×512).

**Request:**
```json
{ "action": "paste_image", "preprocess": true }
```

**Response:**
```json
{ "status": "ok", "image_path": "C:/.../temp/clipboard_ref_image.png" }
```

#### `preprocess_image`
Loads an image from a given path, preprocesses it, and saves the result.

**Parameters:**
- `image_path` (string, required) – path to the source image.

**Request:**
```json
{ "action": "preprocess_image", "image_path": "C:/input.png" }
```

**Response:**
```json
{ "status": "ok", "image_path": "C:/.../temp/preprocessed_ref_image.png" }
```

#### `mesh_from_image`
Generates a 3D mesh (GLB) from a preprocessed image.

**Parameters:**
- `image_path` (string, required)
- `resolution` (string, default "1024") – "512", "1024", "1024_cascade", "1536_cascade"
- `texture_size` (int, default 2048)
- `generate_texture` (bool, default true)
- `remesh` (bool, default false)
- `decimation_target_count` (int, default 100000)
- Sampling parameters: `ss_sampling_steps`, `ss_guidance_strength`, `ss_guidance_rescale`, `ss_rescale_t`, `shape_slat_sampling_steps`, `shape_slat_guidance_strength`, `shape_slat_guidance_rescale`, `shape_slat_rescale_t`, `tex_slat_sampling_steps`, `tex_slat_guidance_strength`, `tex_slat_guidance_rescale`, `tex_slat_rescale_t`

**Request example:**
```json
{
  "action": "mesh_from_image",
  "image_path": "C:/temp/ref.png",
  "resolution": "1024",
  "texture_size": 2048,
  "generate_texture": true
}
```

**Response:**
```json
{ "status": "ok", "data": { "mesh": "C:/temp/mesh.glb" }, "message": "Model generation job completed" }
```

#### `reprocess_last_mesh`
Reapplies post‑processing to the last generated mesh without re‑running generation.

**Parameters:** `generate_texture`, `texture_size`, `remesh`, `decimation_target_count`

**Request:**
```json
{ "action": "reprocess_last_mesh", "decimation_target_count": 50000 }
```

**Response:** same as `mesh_from_image`.

#### `textured_from_mesh_and_image`
Generates a new texture for an existing mesh using a reference image.

**Parameters:**
- `image_path` (string, required)
- `mesh_path` (string, required)
- `texture_size` (int, default 2048)
- `tex_slat_sampling_steps`, `tex_slat_guidance_strength`, `tex_slat_guidance_rescale`, `tex_slat_rescale_t`

**Request:**
```json
{
  "action": "textured_from_mesh_and_image",
  "image_path": "C:/ref.png",
  "mesh_path": "C:/mesh.glb"
}
```

**Response:**
```json
{ "status": "ok", "data": { "mesh": "C:/temp/textured.glb", "new_uv": true } }
```

### Client Example (Python)

```python
import socket, json

def send_job(job):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("127.0.0.1", 50007))
        s.send(json.dumps(job).encode())
        return json.loads(s.recv(4096).decode())

# Paste from clipboard
res = send_job({"action": "paste_image", "preprocess": True})
# Generate mesh
res = send_job({"action": "mesh_from_image", "image_path": res["image_path"]})
print(res["data"]["mesh"])
```