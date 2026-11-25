"""Recover original-channel RMS from independent rotated SFP captures.

The forward transform is y = c H D x; the inverse is x = D H y / (B c).
By default only FFN-down inputs are exported; --all-consumers also covers
RMSNorm consumers. No evaluation labels or logits are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def hadamard(values, block):
    result = values.copy().reshape(len(values), -1, block)
    width = 1
    while width < block:
        pairs = result.reshape(len(values), -1, width * 2)
        left, right = pairs[:, :, :width].copy(), pairs[:, :, width:].copy()
        pairs[:, :, :width], pairs[:, :, width:] = left + right, left - right
        width *= 2
    return result.reshape(values.shape)


def signs_for(k):
    positions = np.arange(k, dtype=np.uint64)
    hashes = (positions * np.uint64(0x9E3779B9) + np.uint64(0x7F4A7C15)) & np.uint64(0xffffffff)
    return 1.0 - 2.0 * (hashes >= np.uint64(0x80000000))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--all-consumers', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_manifest = args.capture.parent / (args.capture.name + '.run.json')
    manifest = {'capture': str(args.capture), 'method': 'uncentered RMS after inverse rotation',
                'consumer': 'all' if args.all_consumers else 'FFN down only',
                'writer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'capture_run_sha256': hashlib.sha256(run_manifest.read_bytes()).hexdigest()
                    if run_manifest.exists() else None,
                'tensors': {}}
    pattern = '*.json' if args.all_consumers else 'linear_w_*.json'
    for metadata in sorted(args.capture.glob(pattern)):
        meta = json.loads(metadata.read_text())
        if 'K' not in meta:
            continue
        k, block = meta['K'], meta['rotation_block']
        assert block in (64, 128) and k % block == 0 and meta['hash_bits'] == 32
        assert meta['bf16_rounded_input'] and meta['activation_scale_applied']
        rotated = np.fromfile(metadata.with_suffix('.f32'), dtype='<f4').reshape(-1, k).astype(np.float64)
        assert len(rotated) == meta['rows'] and len(rotated) > 0
        assert np.isfinite(rotated).all()
        normalization = float(np.float32(0.125 if block == 64 else 0.08838834764831845))
        signs = signs_for(k)
        original = hadamard(rotated, block) * signs / (block * normalization)
        # A complete inverse/forward check catches sign placement and indexing
        # mistakes; H and D do not commute.
        reconstructed = hadamard(original[:4] * signs, block) * normalization
        np.testing.assert_allclose(reconstructed, rotated[:4], rtol=1e-12, atol=1e-12)
        rms = np.sqrt(np.mean(original * original, axis=0)).astype('<f4')
        assert np.isfinite(rms).all() and (rms >= 0).all()
        destination = args.output / (metadata.stem + '.rms')
        with destination.open('wb') as stream:
            stream.write(struct.pack('<8sQ', b'MMI8RM01', k))
            stream.write(rms.tobytes())
        manifest['tensors'][metadata.stem] = {
            'K': k, 'rows': len(rotated),
            'metadata_sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
            'capture_sha256': hashlib.sha256(metadata.with_suffix('.f32').read_bytes()).hexdigest(),
            'rms_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'rms_min_median_p95_max': np.percentile(rms, [0, 50, 95, 100]).tolist()}
    if not manifest['tensors']:
        raise ValueError('no matching activation captures found')
    # A shared normalized input must have one common channel scale for all
    # consumers. Reject inconsistent captures before a model can load them.
    if args.all_consumers:
        for name in manifest['tensors']:
            peer = None
            if name.startswith('gating1_w_'):
                peer = name.replace('gating1_w_', 'gating2_w_', 1)
            elif name.startswith('qkv1_w_'):
                peer = name.replace('qkv1_w_', 'qkv2_w_', 1)
            if peer is not None:
                if peer not in manifest['tensors']:
                    raise ValueError(f'missing shared-input consumer {peer}')
                if (args.output / (name + '.rms')).read_bytes() != (args.output / (peer + '.rms')).read_bytes():
                    raise ValueError(f'shared consumers have different RMS: {name}, {peer}')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'tensors': len(manifest['tensors'])}), flush=True)


if __name__ == '__main__':
    main()
