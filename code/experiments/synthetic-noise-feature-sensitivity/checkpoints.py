"""Single terminal Adam-resumable E-CH4-14 checkpoint, never an auto-resume API."""
from __future__ import annotations
from hashlib import sha256
import io
import json
from pathlib import Path
import torch


def _sha(data):
    return sha256(data).hexdigest()


def save_checkpoint(path, logits, optimizer, step, metadata, *, fixture=False):
    """Create one step-30 checkpoint and detached checksum; refuse all overwrites.

    Only state_dict tensor and JSON-compatible data are serialized. The loader
    insists on `weights_only=True` and a separately trusted expected SHA.
    """
    if step != 30:
        raise ValueError('Only terminal step 30 may be checkpointed')
    if not fixture:
        required={'subset_id','family','data_sha256','validation_sha256',
                  'feature_mapping','gamma','objective','tau','beta','lambda',
                  'H','lr','source_commit','trace_sha256','test_accessed'}
        if required - set(metadata) or metadata.get('test_accessed') is not False:
            raise ValueError('Incomplete production checkpoint provenance')
        if metadata.get('objective') != 'softplus_JS_tau005':
            raise ValueError('Unapproved objective')
    path=Path(path)
    digest_path=path.with_suffix(path.suffix+'.sha256')
    if path.exists() or digest_path.exists():
        raise FileExistsError('Final checkpoint immutable')
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    state=optimizer.state_dict()
    if len(state['param_groups'])!=1 or len(state['param_groups'][0]['params'])!=1:
        raise ValueError('Expected one Adam policy parameter')
    param_id=state['param_groups'][0]['params'][0]
    if param_id not in state['state']:
        raise ValueError('Adam state not initialized')
    adam_step=state['state'][param_id]['step']
    if int(adam_step) != step:
        raise ValueError('Checkpoint Adam update counter mismatch')
    payload={'schema':'e-ch4-14-resumable-v1','step':step,
             'logits':logits.detach().clone().to('cpu'),
             'optimizer':state,'metadata':metadata}
    buffer=io.BytesIO()
    torch.save(payload,buffer)
    raw=buffer.getvalue()
    with path.open('xb') as stream:
        stream.write(raw)
    with digest_path.open('x',encoding='ascii') as stream:
        stream.write(_sha(raw)+'\n')
    return _sha(raw)


def load_checkpoint(path, *, expected_sha256, expected_metadata=None):
    """Load only a trusted, SHA-pinned checkpoint. Never starts training."""
    path=Path(path)
    if not isinstance(expected_sha256,str) or len(expected_sha256)!=64:
        raise PermissionError('Expected SHA must come from trusted manifest')
    raw=path.read_bytes()
    if _sha(raw)!=expected_sha256:
        raise ValueError('Checkpoint SHA mismatch')
    sidecar=path.with_suffix(path.suffix+'.sha256')
    if sidecar.read_text(encoding='ascii').strip()!=expected_sha256:
        raise ValueError('Checkpoint checksum sidecar mismatch')
    payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if payload.get('schema')!='e-ch4-14-resumable-v1' or payload.get('step')!=30:
        raise ValueError('Unexpected checkpoint schema or step')
    if expected_metadata is not None and payload.get('metadata')!=expected_metadata:
        raise ValueError('Checkpoint source metadata mismatch')
    if not isinstance(payload.get('logits'),torch.Tensor) or payload['logits'].dtype!=torch.float64:
        raise ValueError('Checkpoint float64 logits required')
    state=payload.get('optimizer')
    if not isinstance(state,dict) or len(state.get('param_groups',[]))!=1:
        raise ValueError('Optimizer state missing')
    param_id=state['param_groups'][0]['params'][0]
    if int(state['state'][param_id]['step'])!=30:
        raise ValueError('Checkpoint optimizer step mismatch')
    return payload
