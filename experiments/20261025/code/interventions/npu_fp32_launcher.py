"""Set and attest NPU precision before executing an otherwise frozen script.

Usage: python npu_fp32_launcher.py frozen_script.py [script arguments...]
An --output argument supplies the sidecar path unless HET_NPU_RECEIPT is set.
No ACL precision mode is silently selected: this only disables documented HF32
conv/matmul paths and records the effective values in the installed adapter.
"""
import hashlib
import json
import os
import platform
import runpy
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: npu_fp32_launcher.py SCRIPT [ARGS...]')
    target = Path(sys.argv[1]).resolve()
    script_args = sys.argv[2:]
    receipt_env = os.environ.get('HET_NPU_RECEIPT')
    if receipt_env:
        receipt_path = Path(receipt_env)
    elif '--output' in script_args:
        output = Path(script_args[script_args.index('--output') + 1])
        receipt_path = output.with_suffix(output.suffix + '.npu-precision.json')
    else:
        raise SystemExit('Supply script --output or unique HET_NPU_RECEIPT')
    if receipt_path.exists():
        raise FileExistsError('Never overwrite an NPU precision receipt')
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        'kind': 'npu_fp32_launcher_receipt',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'launcher_sha256': digest(__file__), 'target_script': str(target),
        'target_sha256': digest(target), 'script_arguments': script_args,
        'python': sys.version, 'platform': platform.platform(),
        'pid': os.getpid(),
        'environment': {k: os.environ.get(k) for k in [
            'ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME', 'ASCEND_VISIBLE_DEVICES',
            'ASCEND_RT_VISIBLE_DEVICES', 'ACL_PRECISION_MODE',
            'TORCH_NPU_ALLOW_INTERNAL_FORMAT', 'PYTORCH_NPU_ALLOC_CONF']},
        'policy': 'FP32 inputs; disable NPU conv/matmul HF32; no autocast requested by launcher',
        'status': 'initializing',
    }

    def save():
        temporary = receipt_path.with_suffix(receipt_path.suffix + '.tmp')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(receipt_path)

    save()
    try:
        import torch
        import torch_npu
        record['torch'] = torch.__version__
        record['torch_npu'] = getattr(torch_npu, '__version__', 'unavailable')
        record['hf32_before'] = {
            'conv': bool(torch.npu.conv.allow_hf32),
            'matmul': bool(torch.npu.matmul.allow_hf32),
        }
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
        record['hf32_effective'] = {
            'conv': bool(torch.npu.conv.allow_hf32),
            'matmul': bool(torch.npu.matmul.allow_hf32),
        }
        if any(record['hf32_effective'].values()):
            raise RuntimeError('Installed adapter did not retain both HF32=False settings')
        record['hardware'] = str(torch.npu.get_device_name(0))
        record['status'] = 'running'
        save()
        sys.argv = [str(target), *script_args]
        sys.path.insert(0, str(target.parent))
        runpy.run_path(str(target), run_name='__main__')
        record['status'] = 'completed'
    except SystemExit as error:
        record['status'] = 'completed' if error.code in (None, 0) else 'failed'
        record['system_exit_code'] = error.code
        raise
    except BaseException as error:
        record['status'] = 'failed'
        record['exception_type'] = type(error).__name__
        record['exception'] = str(error)
        record['traceback'] = traceback.format_exc()
        raise
    finally:
        record['completed_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
