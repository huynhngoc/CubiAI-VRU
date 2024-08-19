# CubiAI

Deep learning can detect elbow disease in dogs screened for elbow dysplasia
<Link to paper>


## Generate dataset
`dataset_gen/gen_normal_abnormal.py`

Check for duplication
`dataset_gen/QA_data_splitting.py`

## Run experiments locally
```
python experiment_binary.py config/local/pretrain.json path_to_folder/pretrain --temp_folder path_to_temp_folder/pretrain --epochs 20
```


## Run experiemts on Orion
```
sbatch slurm_pretrain_binary.sh config/pretrain/b0_normal_level2.json b0_normal_level2 2
sbatch slurm_pretrain_multiclass.sh config/pretrain/b0_normal_level1_level2.json b0_normal_level1_level2 2

sbatch slurm_scratch_binary.sh config/scratch/b0_normal_level2.json b0_normal_level2 2
sbatch slurm_scratch_multiclass.sh config/scratch/b0_normal_level1_level2.json b0_normal_level1_level2 2
```
