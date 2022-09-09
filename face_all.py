import os
import numpy as np
import cv2


def new_file(path: str):
    """
    创建文件夹
    Args:
        path:想要创建的路径

    Returns:

    """
    if os.path.isdir(path):
        pass
    else:
        print('未找到该文件夹。开始创建文件夹')
        os.makedirs(path)


def get_data_face(ID):
    """
    获取人脸数据集并保存到npy文件中
    :param ID: 当前需要检测的人脸的ID，默认输入姓名首字母缩写
    :return:
    """
    """
    :return:
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 打开摄像头

    front_face = 'face.xml'  # 人脸模型路径
    faceCascade = cv2.CascadeClassifier(front_face)  # 导入人脸检测分类器

    # ID = 'lmq'  # 当前需要检测的人脸的ID，默认输入姓名首字母缩写
    ID = ID
    # save_path = './picture/' + ID + '/'  # 保存数据集图片的路径
    save_path = os.path.join('./picture/', ID)
    # npy_path = './my_model/resource/' + ID + '.npy'  # 保存npy文件的路径，这里的npy装的是图片的数组
    npy_path = os.path.join('./my_model/resource/', ID+'.npy')
    print(save_path, npy_path)
    new_file(save_path)  # 如果文件夹不存在，则创建文件夹
    i = 0  # 计数用，用来拍摄多少张数据集
    images = []  # 存储图片的数组
    while True:
        if i == 50:  # 默认只拍摄50张当数据集
            cv2.destroyAllWindows()  # 关闭窗口
            cap.release()  # 释放摄像头
            np.save(npy_path, images)  # 保存图片数据到npy文件
            break
        _, img = cap.read()  # 获取图片
        if _:
            img = cv2.flip(img, 1)  # 镜像翻转图片
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图片

            # 进行人脸多分类检测
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(5, 5)
            )

            # print(faces)
            print("发现{0}个人脸！".format(len(faces)))
            for (x, y, w, h) in faces:  # 对检测到的人脸中的所有坐标
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 画出人脸框

                if len(faces) == 1:  # 因为一次只录入一个人脸，所以固定当视野范围内只有一个人脸时才进行操作
                    # filename = save_path + ID + '_' + str(i) + '.jpg'  # 保存的图片名称
                    filename = os.path.join(save_path, ID + '_' + str(i) + '.jpg')
                    # print(filename)
                    cv2.imshow('other', img[y:y + h, x:x + w])  # 展示获取的人脸
                    cv2.imwrite(filename, gray[y:y + h, x:x + w])  # 保存截取的人脸图片
                    images.append(gray[y:y + h, x:x + w])  # 累积得到的人脸数据
                    i += 1  # 累加，用于更迭 保存的图片名称
                    break

                # 用于判断方框的四个点
                # cv2.circle(img, (x, y), 3, (255, 0, 0), 3)
                # cv2.circle(img, (x+w, y), 3, (255, 255, 0), 3)
                # cv2.circle(img, (x+w, y+h), 3, (255, 255, 255), 3)
                # cv2.circle(img, (x, y+h), 3, (0, 0, 255), 3)

            cv2.imshow('img', img)
            #
            cv2.waitKey(1)


def face_train(rec: str="L"):
    """
    对人脸数据进行训练
    :param rec: 选择是哪个算法，"L":LBPH, "E":EigenFaces, "F":FisherFaces
    :return:
    """
    rec = rec

    path = 'my_model/resource/'  # 人脸数据集所在路径
    file = os.listdir(path)  # 获取所有npy的名称，如 lmq.npy
    # print(file)
    images = []  # 获取npy的人脸
    names = {}  # 获取对应的人的标签，从0开始
    labels = []  # 获取人脸数据集的数量，从0开始
    label_i = 0  # 用于迭代人的标签

    copy_images = []  # 用来复制用的数组，记录 EigenFaces和FisherFaces 识别器的数据集

    names_path = './LBPH/names/1.npy'  # 人脸标签保存的路径
    model_name = './LBPH/model/1.yml'  # 训练后模型保存的路径

    for j in file:
        file_path = path + j  # 完整的npy路径
        faces = np.load(file_path, allow_pickle='TRUE')  # 读取单个文件的人脸数据集
        names.setdefault(label_i, j.split('.')[0])  # 读取该文件的人脸ID
        for i in faces:  # 读取该文件内的所有人脸图片
            images.append(i)
            labels.append(label_i)
        label_i += 1  # 下一个文件
    np.save(names_path, names)  # 保存人脸标签

    if rec != "L":
        for img in images:
            copy = cv2.resize(img, (200, 200))  # EigenFaces和FisherFaces 识别器需要输入大小相同的图片
            cv2.imshow('copy', copy)
            cv2.waitKey(1)
            copy_images.append(copy)

    # print(len(images), images)
    #
    # print(len(copy_images), copy_images)
    # print(len(names), names)
    # print(len(labels), labels)


    if rec == 'L':
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # 定义人脸识别器
        recognizer.train(images, np.array(labels))  # 训练
    elif rec == 'E':
        recognizer = cv2.face.EigenFaceRecognizer_create()
        recognizer.train(copy_images, np.array(labels))
    elif rec == 'F':
        recognizer = cv2.face.FisherFaceRecognizer_create()
        recognizer.train(copy_images, np.array(labels))

    recognizer.write(model_name)  # 保存


def face_recognize(rec:str="F"):
    """
    实现人脸识别的实时推理
    :return:
    """
    rec = rec  # 选择哪种人脸识别方法
    names_path = './F_Faces/names/1.npy'  # 人脸标签路径
    model_name = './F_Faces/model/1.yml'  # 人脸识别模型路径

    names = np.load(names_path, allow_pickle='TRUE').item()  # 导入人脸标签
    if rec == 'L':
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # 定义人脸识别器
    elif rec == 'E':
        recognizer = cv2.face.EigenFaceRecognizer_create()
    elif rec == 'F':
        recognizer = cv2.face.FisherFaceRecognizer_create()
    recognizer.read(model_name)  # 导入人脸识别模型

    front_face = 'model/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(front_face)  # 导入人脸检测模型

    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        if _:
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(5, 5)
            )

            for (x, y, w, h) in faces:
                # 绘制矩形框
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # 绘制圆框
                cv2.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=1)

                if len(faces) > 0:  # 当存在多个人脸时，同时检测
                    # cv2.imshow('other', img[y:y + h, x:x + w])
                    copy = gray[y:y + h, x:x + w]
                    if rec != 'L':
                        copy = cv2.resize(copy, (200, 200))
                    label_, confidence = recognizer.predict(copy)
                    """
                    ①对于LBPH来说，confidence是置信度评分。用来衡量识别结果与原有模型之间的距离。
                     0表示完全匹配。通常情况下，认为小于50的值是可以接受的。如果该值大于80则认为差别较大。
                    ②对于EigenFaces来说，confidence是返回的置信度评分。用来衡量识别结果与原有模型之间的距离。
                     0表示完全匹配。该参数值通常在0到20000之间，只要低于5000，都被认为是相当可靠的识别结果。
                     注意，这个范围与LBPH的置信度评分值的范围是不同的。
                    ③对于FisherFaces来说，confidence是置信度评分。用来衡量识别结果与原有模型之间的距离。
                     0表示完全匹配。该参数值通常在0到20000之间，只要低于5000，都被认为是相当可靠的识别结果。
                     注意，该评分值的范围与EigenFaces方法的评分值范围已知，与LBPH的置信度评分值的范围是不同的。
                    """

                    print("标签label={0}, 置信值confidence={1}， 名字是{2}".format(label_, confidence, names[int(label_)]))
                    cv2.imshow(names[int(label_)], copy)
                    # 绘制文本
                    cv2.putText(img, names[int(label_)],
                                (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 1)
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # break
            cv2.imshow('img', img)
            cv2.waitKey(1)


if __name__ == '__main__':
    ID = 'lmq'
    # get_data_face(ID)
    # face_train()
    face_recognize()