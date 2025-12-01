import os
import cv2
import numpy as np
import utils.utils as utils
from net.inception import InceptionResNetV1
from net.mtcnn import mtcnn


class face_rec():
    def __init__(self):
        #-------------------------#
        #   创建mtcnn的模型
        #   用于检测人脸
        #-------------------------#
        self.mtcnn_model = mtcnn()
        self.threshold = [0.5, 0.6, 0.8]
               
        #-----------------------------------#
        #   载入facenet
        #   将检测到的人脸转化为128维的向量
        #-----------------------------------#
        self.facenet_model = InceptionResNetV1()
        model_path = './model_data/facenet_keras.h5'
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在，请确保该路径下有facenet_keras.h5")
        self.facenet_model.load_weights(model_path)

        #-----------------------------------------------#
        #   对数据库中的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        #-----------------------------------------------#
        self.known_face_encodings = []
        self.known_face_names = []
        face_dataset_path = "face_dataset"
        # 检查人脸数据库目录是否存在
        if not os.path.exists(face_dataset_path):
            os.makedirs(face_dataset_path)
            print(f"已创建人脸数据库目录：{face_dataset_path}，请放入注册人脸图片（格式：姓名.jpg）")
        
        face_list = [f for f in os.listdir(face_dataset_path) if f.endswith((".jpg", ".png", ".jpeg"))]
        if len(face_list) == 0:
            print(f"警告：{face_dataset_path} 目录下无有效人脸图片，请放入注册人脸（格式：姓名.jpg）")
        else:
            for face in face_list:
                name = face.split(".")[0]
                img_path = os.path.join(face_dataset_path, face)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"跳过无效图片：{img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #---------------------#
                #   检测人脸
                #---------------------#
                rectangles = self.mtcnn_model.detectFace(img, self.threshold)
                if len(rectangles) == 0:
                    print(f"图片 {face} 中未检测到人脸，跳过")
                    continue
                #---------------------#
                #   转化成正方形
                #---------------------#
                rectangles = utils.rect2square(np.array(rectangles))
                #-----------------------------------------------#
                #   facenet要传入一个160x160的图片
                #   利用landmark对人脸进行矫正
                #-----------------------------------------------#
                rectangle = rectangles[0]
                landmark = np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])
                crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                crop_img, _ = utils.Alignment_1(crop_img, landmark)
                crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)
                #--------------------------------------------------------------------#
                #   将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
                #--------------------------------------------------------------------#
                face_encoding = utils.calc_128_vec(self.facenet_model, crop_img)
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
            print(f"人脸数据库加载完成：共 {len(self.known_face_names)} 个注册人脸")

    def recognize(self, draw):
        #-----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        #-----------------------------------------------#
        height, width, _ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        #--------------------------------#
        #   检测人脸
        #--------------------------------#
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)

        if len(rectangles) == 0:
            print("未检测到人脸")
            return draw

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles, dtype=np.int32))
        rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)
        rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)

        #-----------------------------------------------#
        #   对检测到的人脸进行编码
        #-----------------------------------------------#
        face_encodings = []
        for rectangle in rectangles:
            #---------------#
            #   截取图像
            #---------------#
            landmark = np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #-----------------------------------------------#
            #   利用人脸关键点进行人脸对齐
            #-----------------------------------------------#
            crop_img, _ = utils.Alignment_1(crop_img, landmark)
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)

            face_encoding = utils.calc_128_vec(self.facenet_model, crop_img)
            face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:
            #-------------------------------------------------------#
            #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
            #-------------------------------------------------------#
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.9)
            name = "Unknown"
            #-------------------------------------------------------#
            #   找出距离最近的人脸（仅当有注册人脸时）
            #-------------------------------------------------------#
            if len(self.known_face_encodings) > 0:
                face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    print(f"匹配成功：{name}（距离：{face_distances[best_match_index]:.3f}）")
                else:
                    print(f"未匹配到注册人脸（最近距离：{face_distances[best_match_index]:.3f}）")
            face_names.append(name)

        rectangles = rectangles[:, 0:4]
        #-----------------------------------------------#
        #   画框和标注姓名
        #-----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            # 绘制矩形框（匹配成功：绿色，未知：红色）
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(draw, (left, top), (right, bottom), color, 2)
            # 绘制姓名背景（避免文字与图片重叠）
            text_y = bottom + 20 if bottom + 20 < height else bottom - 15
            cv2.rectangle(draw, (left, text_y - 25), (left + len(name) * 15, text_y), color, -1)
            # 绘制姓名
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left + 5, text_y - 5), font, 0.6, (255, 255, 255), 2)
        return draw


if __name__ == "__main__":
    #-------------------------#
    #   初始化人脸识别模型
    #-------------------------#
    try:
        face_recognizer = face_rec()
    except Exception as e:
        print(f"模型初始化失败：{e}")
        exit(1)
    
    #-------------------------#
    #   读取本地测试图
    #-------------------------#
    test_img_path = "test_img.jpg"
    if not os.path.exists(test_img_path):
        print(f"未找到测试图：{test_img_path}，请放入测试图片到当前目录")
        exit(1)
    
    draw = cv2.imread(test_img_path)
    if draw is None:
        print(f"无法读取测试图：{test_img_path}，请检查图片格式（支持jpg/png/jpeg）")
        exit(1)
    
    #-------------------------#
    #   执行人脸识别
    #-------------------------#
    print("开始人脸识别...")
    result_img = face_recognizer.recognize(draw)
    
    #-------------------------#
    #   保存结果图
    #-------------------------#
    result_path = "result_img.jpg"
    cv2.imwrite(result_path, result_img)
    print(f"识别完成！结果已保存到：{result_path}")