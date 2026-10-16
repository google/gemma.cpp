"""Full-input reconstruction calibration, using only independent captured text.

Produces the unchanged signed W8 + group-scale representation. GPTQ-style
error feedback couples columns across all groups. Optionally first compensate
activation quantization with a ridge-regularized least-squares weight update.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import numpy as np
import torch
from calibration_basis import read_raw_header, transform_capture

p = argparse.ArgumentParser()
p.add_argument('--weights', required=True, type=Path)
p.add_argument('--capture', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--group', required=True, type=int, choices=[32, 64, 128])
p.add_argument('--damp', type=float, default=0.1)
p.add_argument('--device', default='cuda:0')
p.add_argument('--rows', type=int, default=2048)
p.add_argument('--block', type=int, default=128)
p.add_argument('--activation-correction', action='store_true')
p.add_argument('--exclude-f32', action='store_true')
p.add_argument('--include', default='')
p.add_argument('--exclude', default='')
args = p.parse_args()
assert args.damp > 0 and args.rows > 0 and args.block > 0 and args.block % args.group == 0
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
args.output.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
torch.cuda.set_device(device)
manifest = {'method': 'full-K GPTQ error feedback', 'args': vars(args).copy(),
            'writer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device),
            'tensors': {}}
manifest['args'] = {k: str(v) if isinstance(v, Path) else v
                    for k, v in manifest['args'].items()}

for raw in sorted(args.weights.glob('*.f32')):
    name = raw.stem
    if args.include and not any(t in name for t in args.include.split(',')):
        continue
    if args.exclude and any(t in name for t in args.exclude.split(',')):
        continue
    metadata_path = args.capture / (name + '.json')
    if not metadata_path.exists():
        continue
    metadata = json.loads(metadata_path.read_text())
    assert metadata['dtype'] == 'float32_le' and metadata['bf16_rounded_input'] is True
    assert metadata['activation_scale_applied'] is True
    if args.exclude_f32 and metadata['source_type'] == 'f32':
        continue
    start = time.monotonic()
    raw_info = read_raw_header(raw)
    n, k = raw_info['N'], raw_info['K']
    rotation, hash_bits = raw_info['rotation'], raw_info['hash_bits']
    assert k % args.group == 0 and metadata['K'] == k
    assert metadata['rotation_block'] == rotation and metadata['hash_bits'] == hash_bits
    wmap = np.memmap(raw, dtype='<f4', mode='r', offset=raw_info['offset'], shape=(n, k))
    capture_path = args.capture / (name + '.f32')
    xcpu = np.fromfile(capture_path, dtype='<f4').reshape(-1, k)
    assert len(xcpu) == metadata['rows'] and len(xcpu) > 0 and np.isfinite(xcpu).all()
    basis_transform = None
    if raw_info['basis'] is not None:
        xcpu, basis_transform = transform_capture(
            xcpu, raw_info['basis'], rotation, hash_bits)
    # Reconstruct exactly the F32 scale and nearest-even int8 activation rule.
    shaped = xcpu.reshape(len(xcpu), -1, args.group)
    amax = np.max(np.abs(shaped), axis=2, keepdims=True)
    inv = np.divide(np.float32(127), amax, out=np.zeros_like(amax), where=amax != 0)
    xhatcpu = (np.rint(shaped * inv).clip(-127, 127) *
               (amax / np.float32(127))).reshape(-1, k)
    x = torch.from_numpy(xcpu).to(device=device, dtype=torch.float64)
    xhat = torch.from_numpy(xhatcpu).to(device=device, dtype=torch.float64)
    h = xhat.T @ xhat
    ridge = args.damp * torch.diag(h).mean().item()
    if ridge == 0:
        raise RuntimeError(f'{name}: all-zero calibration inputs')
    h.diagonal().add_(ridge)
    # FP64 factorization protects the low-rank calibration covariance.
    chol = torch.linalg.cholesky(h)
    hinv = torch.cholesky_inverse(chol)
    u = torch.linalg.cholesky(hinv, upper=True).float()
    del h, chol, hinv
    basis = None
    if args.activation_correction:
        small = xhat @ xhat.T
        small.diagonal().add_(ridge)
        basis = torch.linalg.solve(small, xhat).float()
        del small
    x = x.float()
    xhat = xhat.float()
    residual_input = (x - xhat).T.contiguous()
    output = args.output / (name + '.wq')
    output_magic = b'MMI8WQ02' if raw_info['basis'] is not None else b'MMI8WQ01'
    header = struct.pack('<8sQQQQQ', output_magic, n, k, args.group, rotation, hash_bits)
    basis_bytes = raw_info['basis_bytes']
    q_offset = len(header) + len(basis_bytes)
    with output.open('wb') as f:
        f.write(header)
        f.write(basis_bytes)  # Preserve the exported F32 basis byte for byte.
        f.truncate(q_offset + n * k + (k // args.group) * n * 4)
    qmap = np.memmap(output, dtype=np.int8, mode='r+', offset=q_offset, shape=(n, k))
    smap = np.memmap(output, dtype='<f4', mode='r+', offset=q_offset + n * k,
                    shape=(k // args.group, n))
    baseline_error = final_error = target_energy = 0.0
    for first in range(0, n, args.rows):
        end = min(n, first + args.rows)
        weight_rows = np.array(wmap[first:end])
        if not np.isfinite(weight_rows).all():
            raise ValueError(f'{name}: nonfinite raw weights')
        original = torch.from_numpy(weight_rows).to(device)
        target = original @ x.T
        # Baseline groupwise max-abs quantizer for a direct layer-output check.
        wg = original.reshape(end-first, -1, args.group)
        max0 = wg.abs().amax(dim=2, keepdim=True)
        scales0 = torch.where(max0 == 0, torch.ones_like(max0), max0 / 127)
        inv0 = torch.where(max0 == 0, torch.zeros_like(max0), 127 / max0)
        baseline = (torch.round(wg * inv0).clamp(-127, 127) * scales0).reshape(end-first, k)
        baseline_error += (target - baseline @ xhat.T).double().square().sum().item()
        target_energy += target.double().square().sum().item()
        w = original.clone()
        if basis is not None:
            w.add_((original @ residual_input) @ basis)
        quantized = torch.empty_like(w, dtype=torch.int8)
        group_scales = torch.empty((end-first, k // args.group), device=device)
        reconstructed = torch.empty_like(w)
        for col in range(0, k, args.block):
            stop = min(k, col + args.block)
            local = w[:, col:stop].clone()
            errors = torch.zeros_like(local)
            for j in range(stop - col):
                index = col + j
                if index % args.group == 0:
                    s = local[:, j:j + args.group].abs().amax(dim=1) / 127
                    s = torch.where(s == 0, torch.ones_like(s), s)
                    group_scales[:, index // args.group] = s
                q = torch.round(local[:, j] / s).clamp(-127, 127)
                dequantized = q * s
                quantized[:, index] = q.to(torch.int8)
                reconstructed[:, index] = dequantized
                error = (local[:, j] - dequantized) / u[index, index]
                local[:, j:].sub_(error[:, None] * u[index, index:stop])
                errors[:, j] = error
            if stop < k:
                w[:, stop:].sub_(errors @ u[col:stop, stop:])
        final_error += (target - reconstructed @ xhat.T).double().square().sum().item()
        qmap[first:end] = quantized.cpu().numpy()
        smap[:, first:end] = group_scales.T.cpu().numpy()
        del original, target, wg, max0, inv0, scales0, baseline, w, quantized, reconstructed, group_scales
    qmap.flush()
    smap.flush()
    del qmap, smap, wmap, x, xhat, u, basis, residual_input
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    with raw.open('rb') as source:
        raw_sha256 = hashlib.file_digest(source, 'sha256').hexdigest()
    record = {'N': n, 'K': k, 'samples': len(xcpu), 'ridge': ridge,
              'baseline_reconstruction_sse': baseline_error,
              'calibrated_reconstruction_sse': final_error,
              'target_energy': target_energy, 'seconds': time.monotonic() - start,
              'raw_weights_sha256': raw_sha256,
              'raw_format': raw_info['magic'].decode('ascii'),
              'output_format': output_magic.decode('ascii'),
              'basis_scale_sha256': hashlib.sha256(basis_bytes).hexdigest() if basis_bytes else None,
              'basis_transform': basis_transform,
              'reconstruction_reference': 'scaled floating layer' if basis_bytes else 'original floating layer',
              'capture_metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest()}
    with capture_path.open('rb') as source:
        record['capture_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
    manifest['tensors'][name] = record
    print(json.dumps({'tensor': name, **record}), flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
