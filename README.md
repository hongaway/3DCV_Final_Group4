
## Training
```
python main.py
```

## Evaluation
```
python main.py --eval --resume path/to/checkpoint
```

## Testing
```
python test.py --resume path/to/checkpoint
```

## Checkpoint link
- Sparse DETR
https://drive.google.com/file/d/1EkhefZQBE4OIndyHeEjgCG4uw0mc5fms/view?usp=sharing
- Deformable DETR
https://drive.google.com/file/d/1OopAagrinGF5_e5dTNv83Dng6qJmJLkn/view?usp=sharing
- T6D
https://drive.google.com/file/d/1m6l1CQ6qk4YkV_2HdRZ8k-0ci9CVSuOQ/view?usp=sharing

## Datasets download
1. Download both YCB-Original and YCBV-BOP from NAS
2. Put it into the model you want to test
 

## Datasets Configuration
```
/ycbv_BOP
    /annotations
        /train.json
        /test.json
    /models
    /models_eval
    /models_fine
    /test
    /train_pbr
    /ycbv

/YCB_Video_Dataset
    /image_sets
    /models
```