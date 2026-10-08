"""Verified scaled-basis file metadata and bounded-precision capture transform."""
import hashlib
from pathlib import Path
import struct
import numpy as np
import torch
from prepare_rms_scaling import hadamard, signs_for


def read_raw_header(path):
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(40)
        if len(header) != 40:
            raise ValueError(f'{path}: truncated raw weight header')
        magic, n, k, rotation, hash_bits = struct.unpack('<8sQQQQ', header)
        if magic not in (b'W8RAW001', b'W8RAW002') or not n or not k:
            raise ValueError(f'{path}: invalid raw weight magic or dimensions')
        scaled = magic == b'W8RAW002'
        offset = 40 + (4 * k if scaled else 0)
        if path.stat().st_size != offset + 4 * n * k:
            raise ValueError(f'{path}: raw weight length mismatch')
        basis_bytes = stream.read(4 * k) if scaled else b''
    basis = None
    if scaled:
        if rotation not in (64, 128) or k % rotation or hash_bits != 32:
            raise ValueError(f'{path}: scaled basis supports rotation64/128 and hash32 only')
        basis = np.frombuffer(basis_bytes, dtype='<f4').copy()
        if len(basis) != k or not np.isfinite(basis).all() or not (basis > 0).all():
            raise ValueError(f'{path}: input basis must contain K positive finite float32 values')
    return {'magic': magic, 'N': n, 'K': k, 'rotation': rotation,
            'hash_bits': hash_bits, 'offset': offset,
            'basis_bytes': basis_bytes, 'basis': basis}


def round_bf16(values):
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def forward_rotate(values, rotation):
    values = np.asarray(values, dtype=np.float32)
    if rotation not in (64, 128) or values.shape[1] % rotation:
        raise ValueError('unsupported rotation shape')
    normalization = np.float32(.125 if rotation == 64 else .08838834764831845)
    signs = signs_for(values.shape[1]).astype(np.float32)
    # Explicit sign, butterfly additions/subtractions, then F32 normalization.
    return hadamard(values * signs, rotation) * normalization


def transform_capture(rotated, input_scale, rotation, hash_bits):
    """Recover teacher BF16 channels, scale/BF16-round, then rotate in F32.

    Inverting rounded F32 rotation is approximate, including near-zero channels.
    Returned diagnostics quantify that error; exact zero recovery is not assumed.
    Capture applies A.Scale after rotation but does not serialize its value.
    This transform assumes A.Scale == 1, as in current Gemma activation buffers;
    the reconstruction tolerance is not a proof of that assumption.
    """
    if rotation not in (64, 128) or hash_bits != 32:
        raise ValueError('scaled capture transform supports rotation64/128 and hash32 only')
    rotated = np.asarray(rotated, dtype=np.float32)
    input_scale = np.asarray(input_scale, dtype=np.float32)
    if (rotated.ndim != 2 or not len(rotated) or rotated.shape[1] % rotation or
            input_scale.shape != (rotated.shape[1],) or
            not np.isfinite(rotated).all() or not np.isfinite(input_scale).all() or
            not (input_scale > 0).all()):
        raise ValueError('invalid capture or input scale')
    normalization = np.float32(.125 if rotation == 64 else .08838834764831845)
    inverse = (hadamard(rotated.astype(np.float64), rotation) * signs_for(rotated.shape[1]) /
               (rotation * float(normalization)))
    original = round_bf16(inverse.astype(np.float32))
    reconstructed = forward_rotate(original, rotation)
    max_error = float(np.max(np.abs(reconstructed.astype(np.float64) - rotated)))
    max_relative = max_error / max(float(np.max(np.abs(rotated))), 1e-30)
    if max_relative > 2e-6:
        raise ValueError(f'capture does not reconstruct BF16 input within rotation tolerance: {max_relative}')
    scaled_unrounded = original * input_scale
    scaled = round_bf16(scaled_unrounded)
    transformed = forward_rotate(scaled, rotation)
    if not np.isfinite(transformed).all():
        raise ValueError('scaled capture produced nonfinite values')
    diagnostics = {
        'source': 'exact F32 input_scale bytes exported by C++ in W8RAW002',
        'transform': 'inverse H/D in FP64, BF16 rounding, F32 channel scale, BF16 rounding, F32 D/H rotation',
        'recovery_is_bit_exact': False,
        'activation_scale_assumption': 'A.Scale == 1; capture metadata records application but not its numeric value',
        'activation_scale_assumption_source': 'Current Gemma FFN and normalization activation buffers use default unit scale; not generic nonunit support',
        'capture_reconstruction_max_absolute_error': max_error,
        'capture_reconstruction_max_relative_error': max_relative,
        'inverse_bf16_rounding_max_absolute_error': float(np.max(np.abs(inverse - original))),
        'scaled_bf16_rounding_max_absolute_error': float(np.max(np.abs(scaled_unrounded.astype(np.float64) - scaled))),
        'rotation_normalization_float32': float(normalization),
        'rotation_norm_squared_times_block_minus_one': rotation * float(normalization)**2 - 1.,
        'input_scale_min_max': [float(input_scale.min()), float(input_scale.max())],
        'limitation': 'Isolated teacher-input rescaling; upstream gate/up errors and continuous-scale fold rounding require full-model validation',
    }
    return transformed, diagnostics
