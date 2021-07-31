# yolox_opencv_python
This is a project to deploy YOLOX on Atlas200DK using CANN.
  
在Atlas200dk中使用CANN部署yolox模型推理   
  
# Environments
Firstly, You should have set up the CANN environments on Atlas200DK,    
some other needed packages are as belows   
```
opencv_python (>=4.3 only for opencv dnn inference)  
opencv_contrib_python>=4.3 (only for opencv dnn inference) 
numpy
pyACL (CANN environments have set this)
```

# Usage
##First, git clone this code, yolox_nano.onnx has been on the 'model' dir    

if you want other models, you can download them on the origin repo: https://github.com/Megvii-BaseDetection/YOLOX.git   

and put the downloaded onnx into the ./model dir:   
```
git clone https://github.com/stunback/yolox_for_cann_atlas200dk.git
# if you have downloaded yolox_s.onnx
cd yolox_for_cann_atlas200dk
mv onnx_path model/
```

##Second, remove the focus layer on the onnx model    
change the ONNX_MODEL_PATH on  ./script/yolo_onnx_opt.py   

then run the script:    
```
cd script
python yolo_onnx_opt.py
```

##Third, use atc tool to export the onnx model into cann model
Use yolox_nano_simple.onnx for example:  
```
cd ../model
atc --model=./yolox_nano_simple.onnx --framework=5 --output=yolox_nano_simple --input_format=NCHW --soc_version=Ascend310
```

##At last, run the inference demo
change the model path on src/acl_yolox.py, and run:   
```
cd ../src
python acl_yolox.py 
```

##Additional
An opencv inference demo is also provided:    
```
cd ../src
python main_yolox.py
```

# Model Inference Speed
Hardware:      Atlas200dk npu    
yolox_nano(416)  onnx=308.2ms       cann=11.5ms    

yolox_tiny(416)    onnx=763.8ms       cann=12.2ms    

yolox_s(640)        onnx=2907.3ms     cann=16.5ms    

yolox_x(640)        onnx=24268ms      cann=62.8ms   (4.49GFLOPs/s)    

# Others
Blogs about yolov5, yolox and nanodet:  
https://blog.csdn.net/qq_41035283/article/details/119150751  