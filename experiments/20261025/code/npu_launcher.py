"""Set and report explicit Ascend FP32 policy before a frozen research script."""
import hashlib
import json
import os
import platform
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_npu

torch.npu.conv.allow_hf32 = False
torch.npu.matmul.allow_hf32 = False
policy = {
    'utc': datetime.now(timezone.utc).isoformat(),
    'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
    'python': platform.python_version(), 'architecture': platform.machine(),
    'device': torch.npu.get_device_name(0),
    'conv_allow_hf32': bool(torch.npu.conv.allow_hf32),
    'matmul_allow_hf32': bool(torch.npu.matmul.allow_hf32),
    'synchronization': 'torch.npu.synchronize after timed calls',
    'visible_device_count': torch.npu.device_count(),
    'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
assert not policy['conv_allow_hf32'] and not policy['matmul_allow_hf32']
target = Path(sys.argv[1]).resolve()
policy['script'] = str(target)
policy['script_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
policy['args'] = sys.argv[2:]
receipt_dir = Path(os.environ.get('HET_NPU_RECEIPTS', '/root/het-rebuttal-20261025/launch-receipts'))
receipt_dir.mkdir(parents=True, exist_ok=True)
receipt = receipt_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
receipt.write_text(json.dumps(policy, indent=2))
print(json.dumps({'npu_fp32_launch': policy}), file=sys.stderr, flush=True)
sys.argv = sys.argv[1:]
sys.path.insert(0, str(target.parent))
runpy.run_path(str(target), run_name='__main__')
