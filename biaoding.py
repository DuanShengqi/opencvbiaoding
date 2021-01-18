# _*_ coding:utf-8 _*_

import cv2
import numpy as np
import glob

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((6 * 9, 3), np.float32)

objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("raw_image/*.jpg")
i=0;
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    #print(corners)

    if ret:

        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        #print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (9, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        i+=1;
        cv2.imwrite('detect_img/conimg'+str(i)+'.jpg', img)
        cv2.waitKey(10)

print(len(img_points))
cv2.destroyAllWindows()

# 标定

# print('obj_point:',obj_points)
# print('img_point:',img_points)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

# print("ret:", ret, np.shape(ret))
# print("mtx:\n", mtx, np.shape(mtx)) # 内参数矩阵
# print("dist:\n", dist, np.shape(dist))  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# print("rvecs:\n", rvecs, np.shape(rvecs))  # 旋转向量  # 外参数
# print("tvecs:\n", tvecs ,np.shape(tvecs)) # 平移向量  # 外参数

# print("-----------------------------------------------------")

img = cv2.imread(images[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）

dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
x,y,w,h = roi
dst1 = dst[y:y+h,x:x+w]
cv2.imwrite('result_img/calibresult.jpg', dst1)

# print(rvecs[0], np.shape(rvecs[0]), type(rvecs[0]))
R_rvecs = cv2.Rodrigues(rvecs[0])[0]
# print(R_rvecs, np.shape(R_rvecs))

m_bd = np.matrix(mtx)
R_bd = np.matrix(R_rvecs)
T_bd = np.matrix(tvecs[0])
RT_bd = np.matrix(np.column_stack((R_bd,T_bd)))

target_point = np.matrix(np.ones((4,1)))
target_point_pre = np.matrix(obj_points[0][0])
target_point_pre = np.reshape(target_point_pre, (3,1))
target_point[0:3, :] = target_point_pre

img_target = m_bd * RT_bd * target_point
print(img_target,np.shape(img_target))
target = [(x / img_target[2]).tolist() for x in img_target ]
target = np.reshape(target, (3,1))

print('predict img_point:\n',target)
print('actual img_point:\n',img_points[0][0])