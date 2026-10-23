"""Atomic checkpoint writes and independently tracked validation criteria."""
import io
import json
import math
import os
from pathlib import Path
import tempfile
import hashlib


MAP50='metrics/mAP50(B)'


def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def inspect_checkpoint(data):
    import torch
    checkpoint=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    metrics=checkpoint['train_metrics'];results=checkpoint['train_results']
    epoch=int(checkpoint['epoch'])+1
    if epoch==0:epoch=int(results['epoch'][-1])  # completed runs have optimizer stripped
    assert epoch>0 and int(results['epoch'][-1])==epoch,'Checkpoint/metric epoch mismatch'
    score=float(metrics[MAP50]);fitness=float(metrics['fitness'])
    assert math.isfinite(score) and math.isfinite(fitness)
    assert abs(float(results[MAP50][-1])-score)<=1e-5,'Embedded CSV mAP50 mismatch'
    assert checkpoint.get('ema') is not None or checkpoint.get('model') is not None
    return dict(epoch=epoch,map50=score,fitness=fitness,sha256=hashlib.sha256(data).hexdigest())


def protect_candidate(directory,data,record,provenance):
    """Call once per validated checkpoint. Never replace a higher criterion."""
    directory=Path(directory)
    updates=[]
    for key,filename in (('map50','best_map50'),('fitness','best_fitness')):
        metadata=directory/(filename+'.json');previous=json.loads(metadata.read_text()) if metadata.exists() else None
        if previous is not None and float(previous[key])>=record[key]:continue
        atomic_bytes(directory/(filename+'.pt'),data)
        atomic_bytes(metadata,(json.dumps(dict(record,criterion=key,provenance=provenance),indent=2)+'\n').encode())
        updates.append(key)
    return updates
