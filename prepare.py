import argparse
import collections
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path
import cv2
import yaml
from rgbt.config import SETTINGS

REPO=Path(__file__).resolve().parent

def write(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8',newline='\n')

def external(path):
    path=Path(path).expanduser().resolve()
    if path.is_relative_to(REPO):raise ValueError('Prepared data must be outside the source repository')
    return path

def labels_from_coco(doc,image,annotations,dataset):
    categories={x['id']:x['name'].lower() for x in doc['categories']}
    names={v:k for k,v in SETTINGS[dataset]['names'].items()};w,h=image['width'],image['height'];lines=[]
    for a in annotations:
        if a.get('ignore') or (dataset=='vtuav' and a.get('iscrowd')):continue
        category=categories[a['category_id']]
        if category not in names:continue
        x,y,bw,bh=map(float,a['bbox'])
        if not all(map(math.isfinite,(x,y,bw,bh))) or bw<=0 or bh<=0 or a.get('area',bw*bh)<=0:continue
        x1,y1,x2,y2=max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)
        if x1>=x2 or y1>=y2:continue
        decimals=8 if dataset=='rgbt' else 10
        values=((x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h)
        lines.append(str(names[category])+' '+' '.join(f'{v:.{decimals}f}' for v in values))
    return lines

def coco_pairs(dataset,source,out):
    with zipfile.ZipFile(source if dataset=='rgbt' else source/'VTUAV_v1.0.zip') as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC check failed')
        for split,n in zip(('train','val'),SETTINGS[dataset]['counts']):
            if dataset=='rgbt':
                suffix=f'{split}_thermal.json';matches=[p for p in z.namelist() if p==suffix or p.endswith('/'+suffix)]
                if len(matches)!=1:raise ValueError('Ambiguous or missing '+suffix)
                prefix=matches[0][:-len(suffix)];doc=json.loads(z.read(matches[0]))
            else:
                prefix='';doc=json.loads((source/f'{split}_ir.json').read_text(encoding='utf-8'))
            if len(doc['images'])!=n:raise ValueError('Official split pair count mismatch')
            anns=collections.defaultdict(list)
            for a in doc['annotations']:anns[a['image_id']].append(a)
            seen=set()
            for i,image in enumerate(doc['images']):
                filename=image['file_name']
                if Path(filename).name!=filename or Path(filename).stem in seen:raise ValueError('Unsafe or colliding image name')
                seen.add(Path(filename).stem)
                for raw,folder in (('visible','images'),('thermal','image')) if dataset=='rgbt' else (('rgb','images'),('ir','image')):
                    member=prefix+f'{split}/{raw}/{filename}' if dataset=='rgbt' else f'{"train" if split=="train" else "test"}/{raw}/{filename}'
                    data=z.read(member)
                    import numpy as np
                    im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                    if im is None or im.shape[:2]!=(image['height'],image['width']):raise ValueError('Image/JSON size mismatch: '+member)
                    target=out/folder/split/(filename if dataset=='rgbt' else Path(filename).stem+'.png');target.parent.mkdir(parents=True,exist_ok=True)
                    if dataset=='rgbt':target.write_bytes(data)
                    else:
                        im=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)
                        ok,encoded=cv2.imencode('.png',im,[cv2.IMWRITE_PNG_COMPRESSION,3])
                        if not ok:raise ValueError('PNG encode failed')
                        target.write_bytes(encoded.tobytes())
                lines=labels_from_coco(doc,image,anns[image['id']],dataset)
                write(out/'labels'/split/(Path(filename).stem+'.txt'),'\n'.join(lines)+ ('\n' if lines else ''))
                if (i+1)%1000==0:print(dataset,split,i+1,flush=True)

def vtsar_pairs(source,out):
    for split,n in zip(('train','val'),SETTINGS['vtsar']['counts']):
        def collect(mode,suffix):
            rows=[p for p in source.rglob('*'+suffix) if split in p.parts and mode in p.parts]
            result={p.stem:p for p in rows}
            if len(result)!=len(rows):raise ValueError('Duplicate VTSaR stems: '+mode)
            return result
        rgb,ir,labels=collect('rgb','.jpg'),collect('ir','.jpg'),collect('labels','.txt')
        if len(rgb)!=n or set(rgb)-set(ir) or set(rgb)-set(labels):raise ValueError('VTSaR matched Crop split invalid')
        print('Ignored non-RGB orphan files',split,len(set(ir)-set(rgb)),len(set(labels)-set(rgb)))
        for stem in sorted(rgb):
            for files,folder in ((rgb,'images'),(ir,'image')):
                im=cv2.imread(str(files[stem]))
                if im is None or im.shape!=(480,640,3):raise ValueError('Unexpected Crop640 image: '+str(files[stem]))
                target=out/folder/split/(stem+'.jpg');target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(files[stem],target)
            converted=[]
            for line in labels[stem].read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():continue
                cls,x,y,w,h=map(float,line.split())
                if cls!=0 or not all(map(math.isfinite,(x,y,w,h))) or not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):raise ValueError('Invalid VTSaR label')
                box=(max(0,x-w/2),max(0,y-h/2),min(1,x+w/2),min(1,y+h/2))
                x1,y1,x2,y2=box
                if x1>=x2 or y1>=y2:raise ValueError('Empty clipped VTSaR box')
                if box==(x-w/2,y-h/2,x+w/2,y+h/2):converted.append(line.strip())
                else:converted.append(f'0 {(x1+x2)/2:.12f} {(y1+y2)/2:.12f} {x2-x1:.12f} {y2-y1:.12f}')
            write(out/'labels'/split/(stem+'.txt'),'\n'.join(converted)+ ('\n' if converted else ''))

def audit(root,names):
    report={};cross={}
    for split in ('train','val'):
        images=sorted(p for p in (root/'images'/split).iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg'))
        ir=list((root/'image'/split).glob('*'));label_files=list((root/'labels'/split).glob('*.txt'))
        if {p.name for p in images}!={p.name for p in ir}:raise ValueError('RGB/IR pairing mismatch')
        if {p.stem for p in images}!={p.stem for p in label_files}:raise ValueError('Image/label pairing mismatch')
        counts=collections.Counter();empty=0
        for path in images:
            thermal=root/'image'/split/path.name;a=cv2.imread(str(path));b=cv2.imread(str(thermal))
            if a is None or b is None or a.shape!=b.shape:raise ValueError('Unreadable or mismatched pair: '+str(path))
            lines=(root/'labels'/split/path.with_suffix('.txt').name).read_text().splitlines();empty+=not bool(lines)
            for line in lines:
                cls,x,y,w,h=map(float,line.split())
                if not cls.is_integer() or int(cls) not in names or not all(map(math.isfinite,(x,y,w,h))) or not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):raise ValueError('Invalid normalized box')
                counts[int(cls)]+=1
            digest=hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest()
            if digest in cross and cross[digest]!=split:raise ValueError('Exact decoded pair overlaps official splits; retain source and investigate')
            cross[digest]=split
        report[split]=dict(pairs=len(images),boxes=dict(counts),empty_images=empty)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=SETTINGS);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args();cv2.setNumThreads(1);spec=SETTINGS[a.dataset]
    if a.audit_only:print(json.dumps(audit(a.source.resolve(),spec['names']),indent=2));return
    if not a.output:p.error('--output is required for conversion')
    out=external(a.output)
    if out.exists():p.error('Output already exists; do not overwrite a prepared dataset')
    out.mkdir(parents=True)
    if a.dataset=='vtsar':vtsar_pairs(a.source.resolve(),out)
    else:coco_pairs(a.dataset,a.source.resolve(),out)
    report=audit(out,spec['names'])
    cfg=dict(path=str(out),train='images/train',val='images/val',test='images/val',
        train_ir='image/train',val_ir='image/val',test_ir='image/val',names=spec['names'])
    write(out/'dataset.yaml',yaml.safe_dump(cfg,sort_keys=False))
    write(out/'preparation.json',json.dumps(dict(dataset=a.dataset,official_split=True,audit=report),indent=2))
    print('READY',out/'dataset.yaml')

if __name__=='__main__':main()
