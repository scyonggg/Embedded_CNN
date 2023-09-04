# Embedded_CNN
### This repository contains:
- MobileNetV1 implemenation via Pytorch.
- Training with ImageNet1K dataset from scratch.
- Evaluate inference result.
   - Measure inference speed and compare to other models (VGG16, GoogleNet)

---

### Model architecture
<img src="https://github.com/scyonggg/Embedded_CNN/assets/77262389/037951e0-6461-4a9f-ae4d-dfb640217656" width="50%" height="50%"/>

### Train from scratch
<img src="https://github.com/scyonggg/Embedded_CNN/assets/77262389/93e07e05-2846-425b-a10a-500390ebd522" width="50%" height="50%"/>

### MobileNet simple implementation in PyTorch
![image](https://github.com/scyonggg/Embedded_CNN/assets/77262389/3be15230-aa73-4e27-8f65-4829030e91a2)

### Evaluate inference speed
- Measure inference time and compare to other models
   - MobileNet is faster for **2x times than GoogleNet** and **6x times than VGG16**
- Refer to `infer_speed.py` for measurement code
<img src="https://github.com/scyonggg/Embedded_CNN/assets/77262389/df5abf1a-f52e-43aa-894c-2f0b1841b303"/>
