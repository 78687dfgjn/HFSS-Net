import yaml
ARCHITECTURE = yaml.safe_load(r"""nc: 3
paper_mode: true
scale: n
scales:
  n: [0.33, 0.25, 1024]
backbone:
  - [-1, 1, "Conv", [64, 3, 2]]
  - [-1, 1, "Conv", [128, 3, 2]]
  - [-1, 3, "C2f", [128, true]]
  - [2, 1, "Conv", [256, 3, 2]]
  - [-4, 1, "Conv", [64, 3, 2]]
  - [-1, 1, "Conv", [128, 3, 2]]
  - [-1, 3, "C2f", [128, true]]
  - [-1, 1, "Conv", [256, 3, 2]]
  - [-3, 1, "RIFusion", [64]]
  - [-4, 6, "C2f", [256, true]]
  - [-1, 1, "Conv", [512, 3, 2]]
  - [-4, 6, "C2f", [256, true]]
  - [-1, 1, "Conv", [512, 3, 2]]
  - [-3, 1, "RIFusion", [128]]
  - [-4, 6, "C2f", [512, true]]
  - [-1, 1, "Conv", [1024, 3, 2]]
  - [-4, 6, "C2f", [512, true]]
  - [-1, 1, "Conv", [1024, 3, 2]]
  - [-3, 1, "RIFusion", [256]]
  - [-4, 3, "C2f", [1024, true]]
  - [-1, 1, "SPPF", [1024, 5]]
  - [-4, 3, "C2f", [1024, true]]
  - [-1, 1, "SPPF", [1024, 5]]
  - [[3, 7, 9, 11], 1, "GPT", [64]]
  - [[10, 12, 14, 16], 1, "GPT", [128]]
  - [[15, 17, 20, 22], 1, "GPT", [256]]
head:
  - [-1, 1, "nn.Upsample", [null, 2, "nearest"]]
  - [[-1, 24], 1, "Concat", [1]]
  - [-1, 3, "C2f", [512]]
  - [-1, 1, "nn.Upsample", [null, 2, "nearest"]]
  - [[-1, 23], 1, "Concat", [1]]
  - [-1, 1, "nn.Identity", []]
  - [30, 3, "C2f", [256]]
  - [-1, 1, "Conv", [256, 3, 2]]
  - [[-1, 28], 1, "Concat", [1]]
  - [-1, 3, "C2f", [512]]
  - [-1, 1, "Conv", [512, 3, 2]]
  - [[-1, 25], 1, "Concat", [1]]
  - [-1, 3, "C2f", [1024]]
  - [[32, 35, 38, 31], 1, "Detect", ["nc"]]
""")
SETTINGS = {
 "rgbt": dict(names={0:"person",1:"rider",2:"crowd"},conf=.25,iou=.30,counts=(4900,1225)),
 "vtuav": dict(names={0:"person"},conf=.25,iou=.50,counts=(11392,5378)),
 "vtsar": dict(names={0:"person"},conf=.03,iou=.45,counts=(4800,533)),
}
