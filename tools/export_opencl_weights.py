"""Exports the TorchScript model for the OpenCL backend.

Writes two files:
  * a weight file with every BatchNorm folded into the convolution before it,
    so the OpenCL network only needs convolution + bias;
  * a reference file with input features and the PyTorch outputs for them,
    used by tests/opencl_tests.cpp to check the OpenCL results.

File layout (little-endian):
  weights:   b"AZCLWGT1", u32 count, then per tensor:
             u32 name_len, name, u32 ndim, u32 dims[ndim], f32 data[]
  reference: b"AZCLREF1", u32 n, f32 features[n*3*15*15],
             f32 policy[n*225] (probabilities), f32 value[n]

Usage:
  python tools/export_opencl_weights.py [model.pt] [weights.bin] [reference.bin]
"""

import struct
import sys

import torch

BOARD_SIZE = 15
CHANNELS = 3
BN_EPS = 1e-5
NUM_RES_BLOCKS = 4
NUM_REFERENCE = 64


def fold(conv_weight, bn):
    """Returns (weight, bias) of conv followed by eval-mode BatchNorm."""
    scale = bn["weight"] / torch.sqrt(bn["running_var"] + BN_EPS)
    weight = conv_weight * scale.view(-1, 1, 1, 1)
    bias = bn["bias"] - bn["running_mean"] * scale
    return weight, bias


def collect(module):
    state = {name: t.detach().float().cpu() for name, t in
             module.state_dict().items()}

    def bn(prefix):
        return {k: state[f"{prefix}.{k}"]
                for k in ("weight", "bias", "running_mean", "running_var")}

    tensors = []
    for i in range(NUM_RES_BLOCKS):
        p = f"res_layers.{i}"
        convs = [("conv1", "bn1"), ("conv2", "bn2")]
        if f"{p}.downsample_conv.weight" in state:
            convs.append(("downsample_conv", "downsample_bn"))
        for conv, norm in convs:
            w, b = fold(state[f"{p}.{conv}.weight"], bn(f"{p}.{norm}"))
            tensors += [(f"{p}.{conv}.weight", w), (f"{p}.{conv}.bias", b)]

    for head in ("p", "v"):
        w, b = fold(state[f"{head}_conv.weight"], bn(f"{head}_bn"))
        tensors += [(f"{head}_conv.weight", w), (f"{head}_conv.bias", b)]
    for fc in ("p_fc", "v_fc1", "v_fc2"):
        tensors += [(f"{fc}.weight", state[f"{fc}.weight"]),
                    (f"{fc}.bias", state[f"{fc}.bias"])]
    return tensors


def write_weights(path, tensors):
    with open(path, "wb") as f:
        f.write(b"AZCLWGT1")
        f.write(struct.pack("<I", len(tensors)))
        for name, t in tensors:
            t = t.contiguous()
            encoded = name.encode()
            f.write(struct.pack("<I", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<I", t.dim()))
            f.write(struct.pack(f"<{t.dim()}I", *t.shape))
            f.write(t.numpy().astype("<f4").tobytes())


def random_features(generator):
    """Random but game-like positions: alternating stones and a last move."""
    features = torch.zeros(NUM_REFERENCE, CHANNELS, BOARD_SIZE * BOARD_SIZE)
    for n in range(NUM_REFERENCE):
        stones = 0 if n == 0 else int(torch.randint(1, 120, (1,),
                                                    generator=generator))
        cells = torch.randperm(BOARD_SIZE * BOARD_SIZE, generator=generator)
        for k in range(stones):
            # The last stone was placed by the opponent of the player to move.
            channel = 1 if (stones - 1 - k) % 2 == 0 else 0
            features[n, channel, cells[k]] = 1.0
        if stones > 0:
            features[n, 2, cells[stones - 1]] = 1.0
    return features.view(NUM_REFERENCE, CHANNELS, BOARD_SIZE, BOARD_SIZE)


def write_reference(path, module):
    features = random_features(torch.Generator().manual_seed(2026))
    with torch.inference_mode():
        log_policy, value = module(features)
    with open(path, "wb") as f:
        f.write(b"AZCLREF1")
        f.write(struct.pack("<I", NUM_REFERENCE))
        for t in (features, log_policy.exp(), value.view(-1)):
            f.write(t.contiguous().numpy().astype("<f4").tobytes())


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model/alpha-zero.pt"
    weights_path = sys.argv[2] if len(sys.argv) > 2 else "model/alpha-zero.bin"
    reference_path = (sys.argv[3] if len(sys.argv) > 3
                      else "model/opencl-reference.bin")

    # CPU float32 keeps the reference free of TF32 rounding.
    module = torch.jit.load(model_path, map_location="cpu").eval()
    tensors = collect(module)
    write_weights(weights_path, tensors)
    write_reference(reference_path, module)
    print(f"wrote {len(tensors)} tensors to {weights_path}")
    print(f"wrote {NUM_REFERENCE} reference positions to {reference_path}")


if __name__ == "__main__":
    main()
