# Copyright 2026 PASUMO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

log_file = Path(__file__).resolve().parent / "log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("Initializing SlatForge Server")
import warnings
warnings.filterwarnings('ignore', module='flex_gemm')
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Failed to find nvdisasm.exe.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Falling back to regular HTTP download.*')

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16"
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'

#MARK: Global Variables
HOST = "127.0.0.1"
PORT = 50007
if getattr(sys, 'frozen', False):
    base_path = Path(sys.executable).parent
else:
    base_path = Path(__file__).resolve().parent
cache_path = base_path / ".cache"
temp_path = base_path / "temp"
mesh_with_voxel = None

if not temp_path.exists():
    temp_path.mkdir(parents=True, exist_ok=True)

os.environ['HF_HUB_CACHE'] = str(cache_path)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

#MARK: Ensure Model Cache
try:
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    import subprocess

    def ensure_hf_model_cache(repo_id, filename, link_name, log_name=None):
        """
        Ensures a Hugging Face model is cached and a symlink is created if needed.
        Returns the resolved model path.
        """
        model_path = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        check_model = False
        if model_path and isinstance(model_path, str):
            model_path = Path(model_path).parent.parent.parent
            model_link = model_path.with_name(link_name)
            if not model_link.exists():
                check_model = True
        if not model_path or check_model:
            logging.info(f"{log_name or repo_id} model not found in cache. Downloading now...")
            model_path = snapshot_download(repo_id)
            model_path = Path(model_path).parent.parent
            model_link = model_path.with_name(link_name)
            if not model_link.exists():
                print("Removing existing junction if it exists...")
                subprocess.call(f'rmdir "{model_link}"', shell=True)
                cmd = 'mklink /J "%s" "%s"' % (model_link, model_path)
                logging.info(f"Creating junction with command: {cmd}")
                subprocess.call(cmd, shell=True)
        return model_path

    dinov3_path = ensure_hf_model_cache(
        repo_id="camenduru/dinov3-vitl16-pretrain-lvd1689m",
        filename="config.json",
        link_name="models--facebook--dinov3-vitl16-pretrain-lvd1689m",
        log_name="DINOv3"
    )
    rmbg_path = ensure_hf_model_cache(
        repo_id="camenduru/RMBG-2.0",
        filename="config.json",
        link_name="models--briaai--RMBG-2.0",
        log_name="RMBG"
    )
    trellis2_path = ensure_hf_model_cache(
        repo_id="microsoft/TRELLIS.2-4B",
        filename="pipeline.json",
        link_name="trellis.2-4b",
        log_name="TRELLIS.2"
    )
    trellis_image_large_path = ensure_hf_model_cache(
        repo_id="microsoft/TRELLIS-image-large",
        filename="pipeline.json",
        link_name="trellis-image-large",
        log_name="TRELLIS-image-large"
    )
except Exception as e:
    logging.error(f"Error during Hugging Face cache check: {e}", exc_info=True)
    raise RuntimeError("Failed to check Hugging Face cache. Please check the logs for details.")

logging.info("All required models are cached and ready.")

#MARK: Import Modules
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, Trellis2TexturingPipeline
    import torch, trimesh, gc, copy, socket, json, time
    from PIL import Image, ImageGrab
    import numpy as np
    import o_voxel.postprocess as o_voxel_postprocess
except Exception as e:
    logging.error(f"Error during module importing: {e}", exc_info=True)
    raise RuntimeError("Failed to import modules. Please check the logs for details.")

#MARK: Initialize Models
logging.info("Initializing Trellis pipelines")
try:
    shape_pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    shape_pipeline.cuda()
    texturing_pipeline = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
    texturing_pipeline.cuda()
except Exception as e:
    logging.error(f"Error during model loading: {e}", exc_info=True)
    raise RuntimeError("Failed to load models. Please check the logs for details.")

#MARK: Job Processing
def process_job(job):
    global shape_pipeline
    global texturing_pipeline
    global mesh_with_voxel
    action = job.get("action", None)
    match job["action"]:
        case None:
            logging.info("Empty job received.")
            return {"status": "bad_request", "message": "No action specified"}
        case "test_connection":
            logging.info("Test connection received.")
            return {"status": "ok", "message": "Connection successful"}
        #MARK: Paste Img
        case "paste_image":
            try:
                clipboard_image = ImageGrab.grabclipboard()
                if isinstance(clipboard_image, Image.Image):
                    logging.info("Paste image from clipboard")
                    if job.get("preprocess", True):
                        logging.info("Preprocessing image")
                        # Preprocess image: crop to content and resize to 512x512
                        clipboard_image = shape_pipeline.preprocess_image(clipboard_image)
                    image_path = temp_path / ("clipboard_ref_image" + '.png')
                    clipboard_image.save(image_path, 'PNG')
                    reset_cuda()
                    return {"status": "ok", "image_path": str(image_path)}
                else:
                    logging.warning("The clipboard doesn't contain an image")
                    return {"status": "bad_request", "message": "The clipboard doesn't contain an image"}
            except Exception as e:
                logging.error(f"Error during pasting image: {e}", exc_info=True)
                reset_cuda()
                return {"status": "error", "message": "An error occurred while pasting the image. Please check the logs for details."}
        case "preprocess_image":
            try:
                if "image_path" not in job:
                    logging.warning("Missing 'image_path' in job data for preprocess_image.")
                    return {"status": "bad_request", "message": "Missing 'image_path' in job data"}
                image = Image.open(job["image_path"])
                image = shape_pipeline.preprocess_image(image)
                output_path = temp_path / "preprocessed_ref_image.png"
                image.save(output_path, 'PNG')
                reset_cuda()
                logging.info(f"Image preprocessed and saved to {output_path}")
                return {"status": "ok", "image_path": str(output_path)}
            except Exception as e:
                logging.error(f"Error during image preprocessing: {e}", exc_info=True)
                reset_cuda()
                return {"status": "error", "message": "An error occurred while preprocessing the image. Please check the logs for details."}
        #MARK: Meshgen
        case "mesh_from_image":
            if "image_path" not in job:
                logging.warning("Missing 'image_path' in job data for mesh_from_image.")
                return {"status": "bad_request", "message": "Missing 'image_path' in job data"}
            mesh_with_voxel = None
            resolution = job.get("resolution", "1024")
            texture_size = int(job.get("texture_size", "2048"))
            ss_sampling_steps = job.get("ss_sampling_steps", 12)
            ss_guidance_strength = job.get("ss_guidance_strength", 7.5)
            ss_guidance_rescale = job.get("ss_guidance_rescale", 0.7)
            ss_rescale_t = job.get("ss_rescale_t", 5.0)
            shape_slat_sampling_steps = job.get("shape_slat_sampling_steps", 12)
            shape_slat_guidance_strength = job.get("shape_slat_guidance_strength", 7.5)
            shape_slat_guidance_rescale = job.get("shape_slat_guidance_rescale", 0.5)
            shape_slat_rescale_t = job.get("shape_slat_rescale_t", 3.0)
            tex_slat_sampling_steps = job.get("tex_slat_sampling_steps", 12)
            tex_slat_guidance_strength = job.get("tex_slat_guidance_strength", 1.0)
            tex_slat_guidance_rescale = job.get("tex_slat_guidance_rescale", 0.0)
            tex_slat_rescale_t = job.get("tex_slat_rescale_t", 3.0)
            decimation_target_count = job.get("decimation_target_count", 100000)
            generate_texture = job.get("generate_texture", True)
            remesh = job.get("remesh", False)
            image_path = job["image_path"]
            
            output_path = temp_path / "mesh.glb"
            try:
                image = Image.open(image_path)
                logging.info(f"Starting mesh generation from image: {image_path}")
                mesh_with_voxel = shape_pipeline.run(
                    image,
                    preprocess_image=False,
                    generate_texture=generate_texture,
                    sparse_structure_sampler_params={
                        "steps": ss_sampling_steps,
                        "guidance_strength": ss_guidance_strength,
                        "guidance_rescale": ss_guidance_rescale,
                        "rescale_t": ss_rescale_t,
                    },
                    shape_slat_sampler_params={
                        "steps": shape_slat_sampling_steps,
                        "guidance_strength": shape_slat_guidance_strength,
                        "guidance_rescale": shape_slat_guidance_rescale,
                        "rescale_t": shape_slat_rescale_t,
                    },
                    tex_slat_sampler_params={
                        "steps": tex_slat_sampling_steps,
                        "guidance_strength": tex_slat_guidance_strength,
                        "guidance_rescale": tex_slat_guidance_rescale,
                        "rescale_t": tex_slat_rescale_t,
                    },
                    pipeline_type={
                        "512": "512",
                        "1024": "1024",
                        "1024_cascade": "1024_cascade",
                        "1536_cascade": "1536_cascade",
                    }[resolution],
                )[0]
                #Save high res mesh before post-processing
                vertices_np : np.ndarray = mesh_with_voxel.vertices.cpu().numpy()
                faces_np : np.ndarray = mesh_with_voxel.faces.cpu().numpy()
                #Save vertices and faces to disk
                np.save(str(temp_path / "high_res_vertices.npy"), vertices_np)
                np.save(str(temp_path / "high_res_faces.npy"), faces_np)
                tmesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
                tmesh.export(str(temp_path / "high_res_mesh.ply"))
                postprocessed_mesh, removed_count = postprocess_mesh(mesh_with_voxel, decimation_target_count, remesh, texture_size, skip_texture=not generate_texture)
                postprocessed_mesh.export(str(output_path), extension_webp=False)
                logging.info(f"Mesh generated and saved to {output_path}")
                reset_cuda()
                return {"status": "ok", "data": {"mesh": str(output_path)}, "message": "Model generation job completed"}
            except KeyboardInterrupt:
                reset_cuda()
                logging.warning("Mesh generation cancelled by user.")
                return {"status": "cancelled", "message": "Mesh generation cancelled by user."}
            except Exception as e:
                logging.error(f"Error during mesh generation: {e}", exc_info=True)
                reset_cuda()
                return {"status": "error", "message": "An error occurred during mesh generation."}
        #MARK: Reprocess Mesh
        case "reprocess_last_mesh":
            if mesh_with_voxel is None:
                logging.warning("No mesh available for reprocessing. Please run Generate Mesh first.")
                return {"status": "bad_request", "message": "No mesh available for reprocessing. Please run Generate Mesh first."}
            generate_texture = job.get("generate_texture", False)
            texture_size = int(job.get("texture_size", "2048"))
            remesh = job.get("remesh", False)
            decimation_target_count = job.get("decimation_target_count", 100000)
            output_path = temp_path / "mesh.glb"
            try:
                logging.info("Reprocessing last mesh.")
                postprocessed_mesh, removed_count = postprocess_mesh(mesh_with_voxel, decimation_target_count, remesh, texture_size, skip_texture=not generate_texture)
                postprocessed_mesh.export(str(output_path), extension_webp=False)
                logging.info(f"Reprocessed mesh saved to {output_path}")
                reset_cuda()
                return {"status": "ok", "data": {"mesh": str(output_path)}, "message": "Postprocessing job completed"}
            except KeyboardInterrupt:
                reset_cuda()
                logging.warning("Mesh processing cancelled by user.")
                return {"status": "cancelled", "message": "Mesh processing cancelled by user."}
            except Exception as e:
                logging.error(f"Error during mesh processing: {e}", exc_info=True)
                reset_cuda()
                return {"status": "error", "message": "An error occurred during mesh processing."}
        #MARK: Texgen
        case "textured_from_mesh_and_image":
            image_path = job.get("image_path", None)
            if not image_path:
                logging.warning("Image path is required for texture generation.")
                return {"status": "bad_request", "message": "Image path is required for texture generation."}
            mesh_path = job.get("mesh_path", None)
            if not mesh_path:
                logging.warning("Mesh path is required for texture generation.")
                return {"status": "bad_request", "message": "Mesh path is required for texture generation."}
            texture_size = int(job.get("texture_size", "2048"))
            tex_slat_sampling_steps = job.get("tex_slat_sampling_steps", 12)
            tex_slat_guidance_strength = job.get("tex_slat_guidance_strength", 1.0)
            tex_slat_guidance_rescale = job.get("tex_slat_guidance_rescale", 0.0)
            tex_slat_rescale_t = job.get("tex_slat_rescale_t", 3.0)
            try:
                image = Image.open(image_path)
                mesh = trimesh.load(mesh_path)
                has_uv = hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
                logging.info(f"Starting texture generation for mesh: {mesh_path} and image: {image_path}")
                output = texturing_pipeline.run(mesh, image, tex_slat_sampler_params={
                    "steps": tex_slat_sampling_steps,
                    "guidance_strength": tex_slat_guidance_strength,
                    "guidance_rescale": tex_slat_guidance_rescale,
                    "rescale_t": tex_slat_rescale_t
                }, texture_size=texture_size)
                output.export(str(temp_path / "textured.glb"), extension_webp=False)
                logging.info(f"Textured mesh saved to {temp_path / 'textured.glb'}")
                reset_cuda()
                return {"status": "ok", "data": {"mesh": str(temp_path / "textured.glb"), "new_uv": not has_uv}, "message": "Texture generation job completed"}
            except KeyboardInterrupt:
                reset_cuda()
                logging.warning("Texture generation cancelled by user.")
                return {"status": "cancelled", "message": "Texture generation cancelled by user."}
            except Exception as e:
                reset_cuda()
                logging.error(f"Error during texture generation: {e}", exc_info=True)
                return {"status": "error", "message": "An error occurred during texture generation."}
            
    logging.warning(f"Unknown or bad request: {job}")
    return {"status": "bad_request"}

#MARK: Utility Functions
def reset_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def postprocess_mesh(mesh, decimation_target_count, remesh, texture_size, skip_texture=False):
    postprocess_args = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "attr_layout": None if skip_texture else mesh.layout,
        "voxel_size": mesh.voxel_size,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": decimation_target_count,
        "texture_size": texture_size,
        "remesh": remesh,
        "remesh_band": 1,
        "remesh_project": 0,
        "verbose": True
    }
    postprocessed_mesh = o_voxel_postprocess.to_glb(**postprocess_args)
    bounds_min = (- 0.6, - 0.6, - 0.6)
    bounds_max = (+ 0.6, + 0.6, + 0.6)
    postprocessed_mesh, removed_count = remove_outside_bbox(postprocessed_mesh, bounds_min, bounds_max)
    if removed_count > 0:
        logging.warning(f"A CUDA error occurred during mesh generation, causing {removed_count} vertices to be removed outside the bounding box. The resulting mesh may have holes and will require manual UV unwrapping before texturing. To avoid this issue, please try again after restarting the worker server.")
    return postprocessed_mesh, removed_count

def remove_outside_bbox(mesh, bounds_min, bounds_max, keep_uvs=False):
    bounds_min = np.asarray(bounds_min)
    bounds_max = np.asarray(bounds_max)
    inside = np.all((mesh.vertices >= bounds_min) & (mesh.vertices <= bounds_max), axis=1)
    keep_verts = np.where(inside)[0]
    removed_count = len(mesh.vertices) - len(keep_verts)
    if removed_count == 0:
        return mesh, 0
    if len(keep_verts) == 0:
        return trimesh.Trimesh(), removed_count
    old_to_new = np.full(len(mesh.vertices), -1, dtype=int)
    old_to_new[keep_verts] = np.arange(len(keep_verts))
    faces = mesh.faces.copy()
    face_mask = np.all(inside[faces], axis=1)
    kept_faces = faces[face_mask]
    new_faces = old_to_new[kept_faces]
    degenerate = (new_faces[:, 0] == new_faces[:, 1]) | \
                 (new_faces[:, 1] == new_faces[:, 2]) | \
                 (new_faces[:, 2] == new_faces[:, 0])
    new_faces = new_faces[~degenerate]
    final_face_mask = face_mask.copy()
    final_face_mask[face_mask] = ~degenerate   # mark only the kept, non-degenerate faces
    new_vertices = mesh.vertices[keep_verts]
    new_visual = None
    if keep_uvs and hasattr(mesh, 'visual') and mesh.visual is not None:
        visual = mesh.visual
        if isinstance(visual, trimesh.visual.TextureVisuals):
            # Create a shallow copy (material is immutable usually)
            new_visual = copy.copy(visual)
            uv = visual.uv
            if uv is not None:
                new_visual.uv = uv[keep_verts]
    kwargs = {
        'vertices': new_vertices,
        'faces': new_faces,
        'process': False  # we handle cleanup ourselves
    }
    if new_visual is not None:
        kwargs['visual'] = new_visual
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uv = mesh.visual.uv
        # Per‑face UVs: shape (n_faces, 3, 2)
        if uv.ndim == 3 and uv.shape[0] == len(mesh.faces) and uv.shape[1] == 3 and uv.shape[2] == 2:
            filtered_uv = uv[final_face_mask]
            kwargs['uv'] = filtered_uv
        # Per‑vertex UVs: shape (n_vertices, 2)
        elif uv.ndim == 2 and uv.shape[0] == len(mesh.vertices) and uv.shape[1] == 2:
            filtered_uv = uv[keep_verts]
            kwargs['uv'] = filtered_uv
        else:
            # Unknown UV format
            pass
    new_mesh = trimesh.Trimesh(**kwargs)
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        new_mesh.visual.material = mesh.visual.material
    return new_mesh, removed_count

#MARK: Main Loop
def main():
    logging.info("Initializing worker server main loop.")
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Python executable: {os.sys.executable}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        logging.info(f"Worker listening on {HOST}:{PORT}")
        logging.info(f"SlatForge Server is ready to receive jobs.")
        try:
            while True:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(4096)
                    if not data:
                        continue
                    job = json.loads(data.decode("utf-8"))
                    result = process_job(job)
                    conn.sendall(json.dumps(result).encode("utf-8"))
                time.sleep(1)
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()