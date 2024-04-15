# Low-power Efficient and Accurate Facial-Landmark Detection for Embedded Systems (ICME 2024 Grand Challenge)

Paper Link: TBD
Abstract: TBD

## Dependencies

* python==3.9.12
* CUDA=12.2
* requirements.txt

## Dataset Preparation


 For Custom dataset: TBD
 
 For ivslab challenge: [Metadata](https://drive.google.com/drive/folders/1w1p6OKh6r4xrkZ66trOuOpdLRzA4qwm9?usp=sharing) 
```script
# the dataset directory:
|-- images/
   |-- ivslab/
      | -- ivslab_facial_train/
      | -- ivslab_facial_test_private_qualification/
   
|-- annotations/
   |-- ivslab/
      | -- train.tsv 
      | -- test.tsv (actually validation set)
      | -- test_q.tsv
      | -- test_q.txt (annotation about the number of faces in per image)
```
For generating metadata, run the following code after setting the image path. 
```shell
cd ./preprocess
python  convert_training_dataset.py 
```

## Usage
* Work directory: set the ${ckpt_dir} in ./conf/alignment.py.
* Pretrained model (Pytorch): ./checkpoints/best_model.pkl




### Teacher Training (take ivslab as example)
##### Note that in this version you need to change the backbone in alignment.py.
```shell
python main.py --mode=train --device_ids=0[,1,2,3] \
               --image_dir=./images --annot_dir=./annotations \
               --data_definition=ivslab --learn_rate=0.0002
```

### Student Training (take ivslab as example)
##### Note that in this version you need to change the teacher's pretrained weight in alignment.py.
```shell
python main.py --mode=train_student --device_ids=0[,1,2,3] \
               --image_dir=./images --annot_dir=./annotations \
               --data_definition=ivslab --learn_rate=0.0002
```
### Testing
```shell
python main.py --mode=test --device_ids=0 \
               --image_dir=${image_dir} --annot_dir=${annot_dir} \
               --data_definition=ivslab \
               --pretrained_weight=${model_path} \
```
 
To test on your own image, the following code could be considered:
```shell
python demo.py
```


## Model Quantization
Our tflite model is at "./best.tflite". 

You can put your best Pytorch checkpoint in ./checkpoints/best_model.pkl and run the following code to convert to tflite model.

```shell
python test_quant.py 
```

For testing on D9300, please Turn OFF NNAPI and run the following code.

output_dir will contain the inference results of the image_list.txt after running the shell below.
```shell
python run_model.py ${image_list.txt} ${output_dir}
```

## Acknowledgments
We sincerely thank the repository listed below. This work could not have been accomplished without their contribution. 

[STAR](https://github.com/ZhenglinZhou/STAR)  [SwinTransformer](https://github.com/ChristophReich1996/Swin-Transformer-V2) 
[MobileViT](https://github.com/apple/ml-cvnets/blob/main/docs/source/en/models/classification/README-mobilevit-v2.md)
