import cv2
import os
import numpy as np
import tensorflow as tf
# from scipy import ndimage

INPUT_DIM = 32
PIXEL_DEPTH = 255
NUM_IMAGES_ALL=1700
NUM_IMAGES_TRAIN=1500
NUM_IMAGES_TEST=300
NUM_IMAGES_VAL = NUM_IMAGES_ALL - NUM_IMAGES_TRAIN
NUM_CLASSES=46
mydict=dict()

def prepData(root_dir):
	class_num=0

	final_train_feats=np.ndarray((NUM_CLASSES*NUM_IMAGES_TRAIN, INPUT_DIM, INPUT_DIM), dtype=np.float32)
	final_train_class=np.ndarray(NUM_CLASSES*NUM_IMAGES_TRAIN, dtype=np.int32)
	final_val_feats=np.ndarray((NUM_CLASSES*NUM_IMAGES_VAL, INPUT_DIM, INPUT_DIM), dtype=np.float32)
	final_val_class=np.ndarray(NUM_CLASSES*NUM_IMAGES_VAL, dtype=np.int32)


	

	start_v, start_t = 0, 0
	end_v, end_t = NUM_IMAGES_VAL, NUM_IMAGES_TRAIN
	end_l = NUM_IMAGES_VAL+NUM_IMAGES_TRAIN


	for cla in os.listdir(root_dir):
		new_path=root_dir + "/" + cla
		mydict[str(cla)]=class_num
		data_point=0
		data_feat = np.ndarray((NUM_IMAGES_ALL, INPUT_DIM, INPUT_DIM), dtype=np.float32)
		#print (cla)
		for file in os.listdir(new_path):
			#print(data_point)
			file_path=new_path + "/" + file
			
			# data_feat[data_point] = ( ndimage.imread(file_path).astype(float) - PIXEL_DEPTH / 2 ) / PIXEL_DEPTH
			data_feat[data_point] = ( cv2.imread(file_path, 0).astype(float) - PIXEL_DEPTH / 2 ) / PIXEL_DEPTH
			
			# norm_image = np.zeros(shape=(INPUT_DIM,INPUT_DIM))
			# norm_image = cv2.normalize(im, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

			data_point+=1

		np.random.shuffle(data_feat)

		valid_letter = data_feat[:NUM_IMAGES_VAL, :, :]
		final_val_feats[start_v:end_v, :, :] = valid_letter
		final_val_class[start_v:end_v] = class_num
		start_v += NUM_IMAGES_VAL
		end_v += NUM_IMAGES_VAL

		train_letter = data_feat[NUM_IMAGES_VAL:end_l, :, :]
		final_train_feats[start_t:end_t, :, :] = train_letter
		final_train_class[start_t:end_t] = class_num
		start_t += NUM_IMAGES_TRAIN
		end_t += NUM_IMAGES_TRAIN

		#print (data_point)
		class_num+=1


	permutation = np.random.permutation(final_train_class.shape[0])
	s_final_train_feats = final_train_feats[permutation,:,:]
	s_final_train_class = final_train_class[permutation]


	permutation = np.random.permutation(final_val_class.shape[0])
	s_final_val_feats = final_val_feats[permutation,:,:]
	s_final_val_class = final_val_class[permutation]

	return (s_final_train_feats, s_final_train_class, s_final_val_feats, s_final_val_class)
	# return (norm_train_feat_np.astype(np.float16), train_class_np, norm_val_feat_np.astype(np.float16), val_class_np)


def prepDataTest(root_dir):
	class_num=0

	final_test_feats=np.ndarray((NUM_CLASSES*NUM_IMAGES_TEST, INPUT_DIM, INPUT_DIM), dtype=np.float32)
	final_test_class=np.ndarray(NUM_CLASSES*NUM_IMAGES_TEST, dtype=np.int32)
	
	start_t = 0
	end_t = NUM_IMAGES_TEST


	for cla in os.listdir(root_dir):
		new_path=root_dir + "/" + cla
		mydict[str(cla)]=class_num
		data_point=0
		data_feat = np.ndarray((NUM_IMAGES_TEST, INPUT_DIM, INPUT_DIM), dtype=np.float32)
		#print (cla)
		for file in os.listdir(new_path):
			#print(data_point)
			file_path=new_path + "/" + file
			
			# data_feat[data_point] = ( ndimage.imread(file_path).astype(float) - PIXEL_DEPTH / 2 ) / PIXEL_DEPTH
			data_feat[data_point] = ( cv2.imread(file_path, 0).astype(float) - PIXEL_DEPTH / 2 ) / PIXEL_DEPTH
			
			# norm_image = np.zeros(shape=(INPUT_DIM,INPUT_DIM))
			# norm_image = cv2.normalize(im, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

			data_point+=1

		np.random.shuffle(data_feat)

		final_test_feats[start_t:end_t, :, :] = data_feat
		final_test_class[start_t:end_t] = class_num
		start_t += NUM_IMAGES_TEST
		end_t += NUM_IMAGES_TEST

		#print (data_point)
		class_num+=1


	permutation = np.random.permutation(final_test_class.shape[0])
	s_final_test_feats = final_test_feats[permutation,:,:]
	s_final_test_class = final_test_class[permutation]

	return (s_final_test_feats, s_final_test_class)
	# return (norm_train_feat_np.astype(np.float16), train_class_np, norm_val_feat_np.astype(np.float16), val_class_np)

if __name__ == "__main__":
	prepData("/exp/abhinav/DevanagariHandwrittenCharacterDataset/Train")
	# run("/exp/abhinav/DevanagariHandwrittenCharacterDataset/Test")