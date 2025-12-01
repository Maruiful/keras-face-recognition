## Face-Recognition：人脸识别算法在Keras当中的实现

**仅用于完成网络安全人脸识别实验，并非原创**

### 目录
1. [所需环境 Environment](#所需环境)
2. [文件下载 Download](#文件下载)
3. [使用方法 Usage](#使用方法)
4. [效果 Performance](#效果)

## 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

## 使用方法
1、先将整个仓库download下来。  
2、下载完之后解压，同时下载facenet_keras.h5文件。  
3、将facenet_keras.h5放入model_data中。  
4、将自己想要识别的人脸放入到face_dataset中。  
5、运行face_recognize.py即可。  
6、align.py可以查看人脸对齐的效果。  

## 效果
face_recognize.py的运行结果：  

