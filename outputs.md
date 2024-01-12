epochs=100 Patience=5:

```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100      2.24G      1.208     0.8777      1.015        125        640: 100%|██████████| 22/22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████
                   all         98        602      0.809      0.758      0.841      0.553

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100      2.12G      1.172     0.8843      1.013         85        640: 100%|██████████| 22/22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████
                   all         98        602      0.841      0.777      0.852       0.56

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100      2.18G      1.153     0.8721      1.004        109        640: 100%|██████████| 22/22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████
                   all         98        602      0.817      0.776      0.835      0.539

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100      2.12G      1.122     0.8665      1.004         95        640: 100%|██████████| 22/22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████
                   all         98        602      0.795      0.711      0.813      0.526

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100      2.17G      1.114     0.8218     0.9784        110        640: 100%|██████████| 22/22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████
                   all         98        602      0.867      0.769      0.864      0.544

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100      2.11G      1.105     0.8285     0.9794         93        640: 100%|██████████| 22/22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████
                   all         98        602      0.794      0.789      0.855       0.55

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100      2.19G      1.123     0.8344     0.9876         50        640: 100%|██████████| 22/22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████
                   all         98        602      0.835      0.729      0.833      0.525
Stopping training early as no improvement observed in last 5 epochs. Best results observed at epoch 27, best model saved as best.pt.
To update EarlyStopping(patience=5) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

32 epochs completed in 0.054 hours.
Optimizer stripped from runs\detect\train21\weights\last.pt, 6.2MB
Optimizer stripped from runs\detect\train21\weights\best.pt, 6.2MB

Validating runs\detect\train21\weights\best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.10.11 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 3070, 8192MiB)
Model summary (fused): 168 layers, 3005843 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████
                   all         98        602      0.843      0.779      0.852      0.561
Speed: 0.2ms preprocess, 2.2ms inference, 0.0ms loss, 2.4ms postprocess per image
```

patience=50 (default) epochs=150:

```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    150/150       2.1G     0.5743       0.36     0.8305         46        640: 100%|██████████| 22/22 [00:02<00:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00
                   all         98        602      0.921      0.908      0.943      0.724

150 epochs completed in 0.497 hours.
Optimizer stripped from runs\detect\train23\weights\last.pt, 6.3MB
Optimizer stripped from runs\detect\train23\weights\best.pt, 6.3MB

Validating runs\detect\train23\weights\best.pt...
Ultralytics YOLOv8.0.227 🚀 Python-3.10.11 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 3070, 8192MiB)
Model summary (fused): 168 layers, 3005843 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00
                   all         98        602      0.925      0.904      0.942      0.727
Speed: 0.2ms preprocess, 1.9ms inference, 0.0ms loss, 1.8ms postprocess per image
```
