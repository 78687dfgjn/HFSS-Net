"""FP32 640-square PTANet-metric evaluation of a release checkpoint."""
import argparse
from pathlib import Path
import rgbt
from rgbt.config import SETTINGS
from rgbt.runtime import evaluate,external,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--dataset',choices=SETTINGS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='0');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--split',choices=['val','test'],default='test')
    a=p.parse_args();spec=SETTINGS[a.dataset];out=external(a.output)
    result=evaluate(a.weights,a.data,a.device,out,spec['conf'],spec['iou'],a.workers,a.split,a.batch)
    write_json(out/'metrics.json',result);print(result)

if __name__=='__main__':main()
