
#----------------------------------------------------------------------------
# 相关模块导入

import numpy as np
import cv2
import dlib
import sys
import os

from scipy.spatial import Delaunay

# 用于一些效果可视化
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# 相关路径设置

ROOT_DIR = os.getcwd()

DATA_PATH = os.path.join(ROOT_DIR, 'data')

RESULT_PATH = os.path.join(DATA_PATH, 'result')

MODEL_PATH = os.path.join(ROOT_DIR, 'model')

DLIB_MODEL_PATH = os.path.join(MODEL_PATH, 'shape_predictor_68_face_landmarks.dat')

try:
	BOTTOM_IMAGE = sys.argv[1]
	MASK_IMAGE = sys.argv[2]
	alpha = sys.argv[3]

except:
	BOTTOM_IMAGE = 'img0.png'
	MASK_IMAGE = 'img1.png'
	alpha = 0.5

#----------------------------------------------------------------------------
# 模型导入

# dlib人脸方框检测, 可以考虑mtcnn
face_detector = dlib.get_frontal_face_detector()

# dlib关键点检测模型(68个), 可以考虑face++(106个), stasm(77个)
shape_predictor = dlib.shape_predictor(DLIB_MODEL_PATH)

#----------------------------------------------------------------------------
# 人脸相关域(dlib)

"""LEFT_FACE = list(range(0, 9)) + list(range(17, 22))
RIGHT_FACE = list(range(9, 17)) + list(range(22, 27))"""
JAW_POINTS = list(range(0, 27))
JAW_END = 17
FACE_END = 68

# cv2.fillConvexPoly多边形画图
OVERLAY_POINTS = [JAW_POINTS]# LEFT_FACE, RIGHT_FACE,

#----------------------------------------------------------------------------
# 68个关键点坐标获取函数

def get_landmarks(img, face_detector = face_detector, shape_predictor = shape_predictor):
	landmarks = face_detector(img, 1)

	return np.matrix([[i.x, i.y] for i in shape_predictor(img, landmarks[0]).parts()])

#----------------------------------------------------------------------------
# 人脸方框区域坐标获取

def face_area_coodinate(img):

	area = face_detector(img, 1)[0]

	return [area.left(), area.top(), area.right(), area.bottom()]

#----------------------------------------------------------------------------
# 读图函数

def imread(filename):

	return cv2.imread(os.path.join(DATA_PATH, filename))

#----------------------------------------------------------------------------
# 图片人脸对齐函数映射计算，输入两个关键点坐标矩阵返回一个对齐关系
# Procrustes 分析法

def transformation_from_points(points1, points2):
	points1 = points1.astype(np.float64)
	points2 = points2.astype(np.float64)
	c1 = np.mean(points1, axis=0)
	c2 = np.mean(points2, axis=0)
	points1 -= c1
	points2 -= c2
	s1 = np.std(points1)
	s2 = np.std(points2)
	points1 /= s1
	points2 /= s2
	U, S, Vt = np.linalg.svd(np.dot(points1.T, points2))
	R = (U * Vt).T
	return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

#----------------------------------------------------------------------------
# 将上面的对齐结果 M 映射到一张图片上
# 该操作会修改mask图的尺寸与bottom图一致并对齐bottom图

def warp_im(im, M, dshape):
	output_im = np.zeros(dshape, dtype=im.dtype)
	cv2.warpAffine(im,
		M[:2],
		(dshape[1], dshape[0]),
		dst=output_im,
		borderMode=cv2.BORDER_TRANSPARENT,
		flags=cv2.WARP_INVERSE_MAP)
	return output_im

#----------------------------------------------------------------------------
# 在特征点上使用 Delaunay 三角剖分

def get_triangles(points):

	return Delaunay(points).simplices

#----------------------------------------------------------------------------
# 仿射变换

def affine_transform(input_image, input_triangle, output_triangle, size):
	warp_matrix = cv2.getAffineTransform(np.float32(input_triangle), np.float32(output_triangle))
	
	output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
	
	return output_image

#----------------------------------------------------------------------------
# 三角变形，Alpha 混合

def morph_triangle(img1, img2, img, tri1, tri2, tri, alpha):
	# 计算三角形的边界框
	rect1 = cv2.boundingRect(np.float32([tri1]))
	rect2 = cv2.boundingRect(np.float32([tri2]))
	rect = cv2.boundingRect(np.float32([tri]))

	tri_rect1 = []
	tri_rect2 = []
	tri_rect_warped = []

	for i in range(0, 3):
		tri_rect_warped.append(
			((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
		tri_rect1.append(
			((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
		tri_rect2.append(
			((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))

	# 在边界框内进行仿射变换
	img1_rect = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
	img2_rect = img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]

	size = (rect[2], rect[3])
	warped_img1 = affine_transform(img1_rect, tri_rect1, tri_rect_warped, size)
	warped_img2 = affine_transform(img2_rect, tri_rect2, tri_rect_warped, size)

	# 加权求和
	img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

	# 生成蒙版
	mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
	cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

	# 应用蒙版
	img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
	img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] * (1 - mask) + img_rect * mask
#----------------------------------------------------------------------------
# 加入图片四个顶点和四条边的中点用于三角剖分

def points_8(image, points):

	x = image.shape[1] - 1
	y = image.shape[0] - 1
	points = points.tolist()
	points.append([0, 0])
	points.append([x // 2, 0])
	points.append([x, 0])
	points.append([x, y // 2])
	points.append([x, y])
	points.append([x // 2, y])
	points.append([0, y])
	points.append([0, y // 2])

	return np.array(points)

#----------------------------------------------------------------------------
# 颜色矫正

def correct_color(img1, img2, landmark):
	blur_amount = 0.4 * np.linalg.norm(
		np.mean(landmark[36:42], axis=0)
		- np.mean(landmark[42:48], axis=0)
		)
	blur_amount = int(blur_amount)

	if blur_amount % 2 == 0:
		blur_amount += 1

	img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
	img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

	img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)

	return img2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)

#----------------------------------------------------------------------------
# 三角融合函数, 本质线性相加：M(x,y)=(1−α)I(x,y)+αJ(x,y)

def morph_face(bottom_img, mask_img, points1, points2, alpha = 0.5):

	points1 = points_8(bottom_img, points1)
	points2 = points_8(mask_img, points2)

	morph_points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)

	bottom_img = np.float32(bottom_img)
	mask_img = np.float32(mask_img)

	img_morphed = np.zeros(bottom_img.shape, dtype = bottom_img.dtype)

	triangles = get_triangles(morph_points)
	for i in triangles:
		x = i[0]
		y = i[1]
		z = i[2]

		tri1 = [points1[x], points1[y], points1[z]]
		tri2 = [points2[x], points2[y], points2[z]]
		tri = [morph_points[x], morph_points[y], morph_points[z]]
		morph_triangle(bottom_img, mask_img, img_morphed, tri1, tri2, tri, alpha)

	return np.uint8(img_morphed)

#----------------------------------------------------------------------------
# opencv泊松融合函数

def merge_img(bottom_img, mask_img, mask_matrix, mask_points, blur_detail_x=None, blur_detail_y=None, mat_multiple=None):
	face_mask = np.zeros(bottom_img.shape, dtype=bottom_img.dtype)

	for group in OVERLAY_POINTS:
		cv2.fillConvexPoly(face_mask, cv2.convexHull(mask_matrix[group]), (255, 255, 255))# 填充人脸多边形
		r = cv2.boundingRect(np.float32([mask_points[:FACE_END]]))
		center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
	# plt.imshow(face_mask)
	# plt.show()
	if mat_multiple:
		mat = cv2.getRotationMatrix2D(center, 0, mat_multiple)
		face_mask = cv2.warpAffine(face_mask, mat, (face_mask.shape[1], face_mask.shape[0]))

	if blur_detail_x and blur_detail_y:
		face_mask = cv2.blur(face_mask, (blur_detail_x, blur_detail_y), center)

	return cv2.seamlessClone(np.uint8(mask_img), bottom_img, face_mask, center, cv2.NORMAL_CLONE)

#----------------------------------------------------------------------------
# 矫正底图

def affine_triangle(src, dst, t_src, t_dst):
	r1 = cv2.boundingRect(np.float32([t_src]))
	r2 = cv2.boundingRect(np.float32([t_dst]))

	t1_rect = []
	t2_rect = []
	t2_rect_int = []

	for i in range(0, 3):
		t1_rect.append((t_src[i][0] - r1[0], t_src[i][1] - r1[1]))
		t2_rect.append((t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]))
		t2_rect_int.append((t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]))

	mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
	cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

	img1_rect = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

	size = (r2[2], r2[3])

	img2_rect = affine_transform(img1_rect, t1_rect, t2_rect, size)
	img2_rect = img2_rect * mask

	dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
	dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

def rect_contains(rect, point):
	if point[0] < rect[0]:
		return False
	elif point[1] < rect[1]:
		return False
	elif point[0] > rect[2]:
		return False
	elif point[1] > rect[3]:
		return False
	return True

def measure_triangle(image, points):
	rect = (0, 0, image.shape[1], image.shape[0])
	sub_div = cv2.Subdiv2D(rect)# 画布
	points = points.tolist()

	for p in points:
		sub_div.insert((p[0],p[1]))# 插入关键点

	triangle_list = sub_div.getTriangleList()# 德劳力三角剖分

	triangle = []
	pt = []

	for t in triangle_list:
		pt.append((t[0], t[1]))
		pt.append((t[2], t[3]))
		pt.append((t[4], t[5]))

		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])

		if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
			ind = []
			for j in range(0, 3):
				for k in range(0, len(points)):
					if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
						ind.append(k)
			if len(ind) == 3:
				triangle.append((ind[0], ind[1], ind[2]))

		pt = []

	return triangle

def tran_src(src_img, src_points, dst_points):
	jaw = JAW_END

	dst_list = points_8(src_img, dst_points)
	src_list = points_8(src_img, src_points)

	jaw_points = []

	for i in range(0, jaw):
		jaw_points.append(dst_list[i].tolist())
		jaw_points.append(src_list[i].tolist())

	warp_jaw = cv2.convexHull(np.array(jaw_points), returnPoints=False)
	warp_jaw = warp_jaw.tolist()
	
	for i in range(0, len(warp_jaw)):
		warp_jaw[i] = warp_jaw[i][0]

	warp_jaw.sort()

	if len(warp_jaw) <= jaw:
		dst_list = dst_list[jaw - len(warp_jaw):]
		src_list = src_list[jaw - len(warp_jaw):]
		for i in range(0, len(warp_jaw)):
			dst_list[i] = jaw_points[int(warp_jaw[i])]
			src_list[i] = jaw_points[int(warp_jaw[i])]
	else:
		for i in range(0, jaw):
			if len(warp_jaw) > jaw and warp_jaw[i] == 2 * i and warp_jaw[i + 1] == 2 * i + 1:
				warp_jaw.remove(2 * i)

			dst_list[i] = jaw_points[int(warp_jaw[i])]

	dt = measure_triangle(src_img, dst_list)

	res_img = np.zeros(src_img.shape, dtype=src_img.dtype)

	for i in range(0, len(dt)):
		t_src = []
		t_dst = []

		for j in range(0, 3):
			t_src.append(src_list[dt[i][j]])
			t_dst.append(dst_list[dt[i][j]])
		affine_triangle(src_img, res_img, t_src, t_dst)

	return res_img

#----------------------------------------------------------------------------
# 操作开始

# 读图
bottom_img = imread(BOTTOM_IMAGE)
mask_img = imread(MASK_IMAGE)

# 获得68个人脸关键点的坐标
landmarks_bottom = get_landmarks(bottom_img)
landmarks_mask= get_landmarks(mask_img)

# 获取的对齐关系
M = transformation_from_points(landmarks_bottom, landmarks_mask)

# 将对齐关系应用到mask图并保存
warped_img = warp_im(mask_img, M, bottom_img.shape)
outfile_path = os.path.join(RESULT_PATH, '1-warped_img_{}_{}.jpg'.format(BOTTOM_IMAGE.split('.')[0], MASK_IMAGE.split('.')[0]))
cv2.imwrite(outfile_path, warped_img)

# 重新定位对齐图
landmarks2_warped = get_landmarks(warped_img)

# 三角变形
morph_img = morph_face(bottom_img, warped_img, landmarks_bottom, landmarks2_warped, float(alpha))
outfile_path = os.path.join(RESULT_PATH, '2-morph_img_{}_{}_{}.jpg'.format(BOTTOM_IMAGE.split('.')[0], MASK_IMAGE.split('.')[0], alpha))
cv2.imwrite(outfile_path, morph_img)

# 裁剪融合图人脸

# 重新定位融合图
landmarks3_morph = get_landmarks(morph_img)

# 矫正融合图脸型与底图一致
tran_morph_img = tran_src(morph_img, landmarks3_morph, landmarks_bottom)
outfile_path = os.path.join(RESULT_PATH, '3-tran_morph_img_{}_{}_{}.jpg'.format(BOTTOM_IMAGE.split('.')[0], MASK_IMAGE.split('.')[0], alpha))
cv2.imwrite(outfile_path, tran_morph_img)

# 修正融合图颜色与底图一致
morph_image_revise = correct_color(bottom_img, morph_img, landmarks_bottom)
outfile_path = os.path.join(RESULT_PATH, '4-morph_img_revise_{}_{}_{}.jpg'.format(BOTTOM_IMAGE.split('.')[0], MASK_IMAGE.split('.')[0], alpha))
cv2.imwrite(outfile_path, morph_image_revise)

# morph_image_revise不是标准的RGB文件保存后重载
morph_image_revise_imread = cv2.imread(outfile_path)

"""# 重新定位修正颜色后的融合图
landmarks4_warped_revise = get_landmarks(tran_bottom_img)"""

# 泊松融合
merged_img = merge_img(bottom_img, morph_image_revise_imread, landmarks_bottom, landmarks_bottom, blur_detail_x = 15, blur_detail_y = 10, mat_multiple = 1.01)
outfile_path = os.path.join(RESULT_PATH, '5-merged_img_{}_{}_{}.jpg'.format(BOTTOM_IMAGE.split('.')[0], MASK_IMAGE.split('.')[0], alpha))
cv2.imwrite(outfile_path, merged_img)

#----------------------------------------------------------------------------