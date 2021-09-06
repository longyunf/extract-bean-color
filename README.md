## Directories
```plain
data/ 	 
	image864/                 		    
	label864/
	msk864/
lib/
scripts/
```

## Code
**1. Data preparation**

```bash
cd scripts

# 1. data augmentation
data_aug.m

# 2. generate patches
gen_patch.m

# 3. split train/val set
split_patch.m
```

**2. Train segmentation model**
```bash
python train.py
```