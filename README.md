# 数据集与模型说明

注：参考文档时可将gsp23a上strtune_dataset中的目录数据拷贝至该repo下，且Binaries、IDBs、Roberta与该repo下各目录在同一层级



## Binaries

模型训练推理所用数据集的binary形式，来源：Marcelli A, Graziano M, Ugarte-Pedrero X, et al. How machine learning is solving the binary function similarity problem[C]//31st USENIX Security Symposium (USENIX Security 22). 2022: 2099-2116.



## ida_scripts

get_idb.py为生成idb文件脚本，microcode/get_mc.py为生成microcode脚本，用法见下



## IDBs

binary通过ida脚本处理得到的.idb或.i64后缀文件。把binary放到Binaries下，生成的idb在IDBs下,数据集的idb存放在IDBs/Dataset-1-new

- **Input**: the flag corresponding to the dataset to process (`--db`)
- **Output**: the corresponding IDBs and a log file (`generate_idbs_log.txt`)

Example: generate the IDBs for the Dataset-1

```c
cd IDA_scripts
python get_idb.py --db1
```

使用ida脚本从idb提取IR，数据集的IR存放在DBs/Dataset-1-new/features

```c
cd microcode
python run.py -i idb_folder -o output_folder
```



## Roberta

涉及数据在gsp23a:/home/kaiyanh/strtune_dataset/Roberta 路径下，需mv至该Roberta目录下

```c
cd Roberta
conda activate sem2vec
python src/run_roberta.py
python src/fine_tune.py
```



## Model

对提取的microcode文件进行切片预处理，例：

```c
cd strand
python gnn_preprocessing_cdg.py -i ../../../DBs/Dataset-1-new/features/testing/acfg_microcode_Dataset-1_testing -d opcodes_dict.json -o ./Dataset-1-new_testing
```

对待计算函数对计算sim结果：

```c
cd GMN
python gnn.py --test -c ./model_checkpoint -o ./Dataset-1-new_testing --featuresdir ../strand -b ../../Roberta/BERT-ft
```



## Results

data下存放不同数据集的不同模型计算结果，即上一步计算得到的csv文件拷贝过来后，需要先通过ipynb删除不需要的列

最后通过res目录中的ipynb计算最终结果

