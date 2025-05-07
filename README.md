# ECG Classification Capstone

## Files

- `methodA.ipynb`: Residual CNNs with Attention Mechanism
- `methodB.ipynb`: Improved ResNet
- `methodC.ipynb`: KNN baseline from Aim 1
- `methodD.py`: Ensemble methodA and methodB
- `ecgLoader.py`: Preprocessing and dataset loading utility

## Dataset

- ECG5000 (OpenML)
- mitbih (Kaggle)

## Results

| Method  | Accuracy | mitbih Accuracy |
| ------- | -------- | --------------- |
| methodA | 99%      | 98%             |
| methodB | 99%      | 98%             |
| methodC | 100%     | 98%             |
| methodD | 100%     | 98%             |
