from hashlib import algorithms_guaranteed
import sys
import getopt
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import boykov_kolmogorov
import ff
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-a','--algo', type=str, required=True)
parser.add_argument('-sp','--sp_en', type=str, required=True)
parser.add_argument('-i','--img', type=str, required=True)
parser.add_argument('-ac','--acc', type=str, required=True)

args = parser.parse_args()
drawing = False
mode = "ob"
marked_ob_pixels = []
marked_bg_pixels = []
I = None
I_dummy = None
l_range = [0, 256]
a_range = [0, 256]
b_range = [0, 256]
lab_bins = [32, 32, 32]

algo = args.algo
sp_en = args.sp_en
acc = args.acc


class SPNode():
	"""docstring for SPNode"""

	def __init__(self):
		self.label = None
		self.pixels = []
		self.mean_intensity = 0.0
		self.centroid = ()
		self.type = 'na'
		self.mean_lab = None
		self.lab_hist = None
		self.real_lab = None

	def __repr__(self):
		return str(self.label)


def mark_seeds(event, x, y, flags, param):
	global drawing, mode, marked_bg_pixels, marked_ob_pixels, I_dummy
	h, w, c = I_dummy.shape

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if mode == "ob":
				if(x >= 0 and x <= w-1) and (y > 0 and y <= h-1):
					marked_ob_pixels.append((y, x))
				cv2.line(I_dummy, (x-3, y), (x+3, y), (0, 0, 255))
			else:
				if(x >= 0 and x <= w-1) and (y > 0 and y <= h-1):
					marked_bg_pixels.append((y, x))
				cv2.line(I_dummy, (x-3, y), (x+3, y), (255, 0, 0))
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == "ob":
			cv2.line(I_dummy, (x-3, y), (x+3, y), (0, 0, 255))
		else:
			cv2.line(I_dummy, (x-3, y), (x+3, y), (255, 0, 0))


def gen_sp_slic(I, region_size_):
	# Superpixel Generation ::  Slic superpixels compared to state-of-the-art superpixel methods
	SLIC = 100
	SLICO = 101
	num_iter = 4
	sp_slic = cv2.ximgproc.createSuperpixelSLIC(
	    I, algorithm=SLICO, region_size=region_size_, ruler=10.0)
	sp_slic.iterate(num_iterations=num_iter)

	return sp_slic


def draw_sp_mask(I, SP):
	I_marked = np.zeros(I.shape)
	I_marked = np.copy(I)
	mask = SP.getLabelContourMask()
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			# SLIC/SLICO marks borders with -1 :: SEED marks borders with 255
			if mask[i][j] == -1 or mask[i][j] == 255:
				I_marked[i][j] = [128, 128, 128]
	return I_marked


def draw_centroids(I, SP_list):
	for each in SP_list:
		if each != None:
			i, j = each.centroid
			I[i][j] = 128
	return I


def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def distance_3d(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2]-p1[2])**2)

# def accuracy(mask)


def gen_graph(I, SP_list, hist_ob, hist_bg):
	G = nx.Graph()
	s = SPNode()
	s.label = 's'
	t = SPNode()
	t.label = 't'
	lambda_ = .9
	sig_ = 5
	hist_ob_sum = int(hist_ob.sum())
	hist_bg_sum = int(hist_bg.sum())

	for u in SP_list:
		K = 0
		region_rad = math.sqrt(len(u.pixels)/math.pi)
		for v in SP_list:
			if u != v:
				if distance(u.centroid, v.centroid) <= 2.5*region_rad:
					sim = math.exp(-(cv2.compareHist(u.lab_hist, v.lab_hist, 3)
					               ** 2/2*sig_**2))*(1/distance(u.centroid, v.centroid))
					K += sim
					G.add_edge(u, v, sim=sim)
		if(u.type == 'na'):
			l_, a_, b_ = [int(x) for x in u.mean_lab]
			# print(lab_bins)
			l_i = int(l_//((l_range[1]-l_range[0])/lab_bins[0]))
			a_i = int(a_//((a_range[1]-a_range[0])/lab_bins[1]))
			b_i = int(b_//((b_range[1]-b_range[0])/lab_bins[2]))
			print(a_i, b_i, l_i, hist_ob[l_i, a_i, b_i])
			pr_ob = int(hist_ob[l_i, a_i, b_i])/hist_ob_sum
			pr_bg = int(hist_bg[l_i, a_i, b_i])/hist_bg_sum
			sim_s = 100000
			sim_t = 100000
			if pr_bg > 0:
				sim_s = lambda_*-np.log(pr_bg)
			if pr_ob > 0:
				sim_t = lambda_*-np.log(pr_ob)
			G.add_edge(s, u, sim=sim_s)
			G.add_edge(t, u, sim=sim_t)
		if(u.type == 'ob'):
			G.add_edge(s, u, sim=1+K)
			G.add_edge(t, u, sim=0)
		if(u.type == 'bg'):
			G.add_edge(s, u, sim=0)
			G.add_edge(t, u, sim=1+K)
	return G


def main():
	global I, mode, I_dummy, algo, sp_en

	inputfile = ''
	# try:
	# 	opts, args = getopt.getopt(sys.argv[1:], "i:ha:s:", [
	# 	                           "input-image=", "help", "algo", "sp"])
	# except getopt.GetoptError:
	# 	print('fast_seg.py -i <input image>')
	# 	sys.exit(2)
	# for opt, arg in opts:
	# 	if opt == '-h':
	# 		print('fast_seg.py -i <input image>')
	# 		sys.exit()
	# 	elif opt in ("-i", "--input-image"):
	# 		inputfile = arg
	# 	elif opt in ("-a", "--algo"):
	# 		global algo 
	# 		algo = arg
	# 	elif opt == '-s':
	# 		global sp_en
	# 		sp_en = arg
	# 		print(sp_en)
	# 		sys.exit()
	inputfile = args.img
	print('Using image: ', inputfile)

	I = cv2.imread(inputfile)  # imread wont rise exceptions by default
	I_dummy = np.zeros(I.shape)
	I_dummy = np.copy(I)

	h, w, c = I.shape
	# global sp_en
	if sp_en == "y":
		region_size = 20
	else:
		region_size = 10

	cv2.namedWindow('Mark the object and background')
	cv2.setMouseCallback('Mark the object and background', mark_seeds)
	while(1):
		cv2.imshow('Mark the object and background', I_dummy)
		k = cv2.waitKey(1) & 0xFF
		if k == ord('o'):
			mode = "ob"
		elif k == ord('b'):
			mode = "bg"
		elif k == 27:
			break
	cv2.destroyAllWindows()

	start = time.process_time()
# your code here    

	I_lab = cv2.cvtColor(I, cv2.COLOR_BGR2Lab)
	SP = gen_sp_slic(I, region_size)
	SP_labels = SP.getLabels()
	SP_list = [None for each in range(SP.getNumberOfSuperpixels())]

	for i in range(h):
		for j in range(w):
			if not SP_list[SP_labels[i][j]]:
				tmp_sp = SPNode()
				tmp_sp.label = SP_labels[i][j]
				tmp_sp.pixels.append((i, j))
				SP_list[SP_labels[i][j]] = tmp_sp
			else:
				SP_list[SP_labels[i][j]].pixels.append((i, j))

	for sp in SP_list:
		if sp != None:
			n_pixels = len(sp.pixels)
			i_sum = 0
			j_sum = 0
			lab_sum = [0, 0, 0]
			tmp_mask = np.zeros((h, w), np.uint8)
			for each in sp.pixels:
				i, j = each
				i_sum += i
				j_sum += j
				lab_sum = [x + y for x, y in zip(lab_sum, I_lab[i][j])]
				# print(lab_sum)
				tmp_mask[i][j] = 255
			sp.lab_hist = cv2.calcHist(
			    [I_lab], [0, 1, 2], tmp_mask, lab_bins, l_range+a_range+b_range)
			sp.centroid += (i_sum//n_pixels, j_sum//n_pixels,)
			sp.mean_lab = [x/n_pixels for x in lab_sum]
			sp.real_lab = [sp.mean_lab[0]*100/255,
			    sp.mean_lab[1]-128, sp.mean_lab[2]-128]

	for pixels in marked_ob_pixels:
		x, y = pixels
		SP_list[SP_labels[x][y]].type = "ob"
	for pixels in marked_bg_pixels:
		x, y = pixels
		SP_list[SP_labels[x][y]].type = "bg"
	I_marked = draw_sp_mask(I, SP)
	I_marked = draw_centroids(I_marked, SP_list)
	mask_ob = np.zeros((h, w), dtype=np.uint8)
	for pixels in marked_ob_pixels:
		i, j = pixels
		mask_ob[i][j] = 255
	mask_bg = np.zeros((h, w), dtype=np.uint8)
	for pixels in marked_bg_pixels:
		i, j = pixels
		mask_bg[i][j] = 255

	hist_ob = cv2.calcHist([I_lab], [0, 1, 2], mask_ob,
	                       lab_bins, l_range+a_range+b_range)

	hist_bg = cv2.calcHist([I_lab], [0, 1, 2], mask_bg,
	                       lab_bins, l_range+a_range+b_range)

	print(hist_bg.shape)
	G = gen_graph(I_lab, SP_list, hist_ob, hist_bg)

	for each in G.nodes():
		if each.label == 's':
			s = each
		if each.label == 't':
			t = each
	# global algo
	if algo == "bk":
		RG = boykov_kolmogorov.boykov_kolmogorov(G, s, t, capacity='sim')
		source_tree, target_tree = RG.graph['trees']
		partition = (set(source_tree), set(G) - set(source_tree))
		sp_label_list = partition[0]
	else:
		sp_label_list = ff.ford_fulkerson(G, s, t)

	F = np.zeros((h, w), dtype=np.uint8)
 
	for sp in sp_label_list:
		for pixels in sp.pixels:
			i,j = pixels 
			F[i][j] = 1
	Final = cv2.bitwise_and(I, I, mask=F)
	print("------------------\n","Time taken : ",time.process_time() - start,"\n------------------")
	global acc 
	if(acc=="y"):
		masked_image = inputfile 
		mask_actual_img = cv2.imread('./images/masks/bunny_mask.png')
		mask_actual = np.zeros((h,w))
		for i in range(h):
			for j in range(w):
				if mask_actual_img[i][j][0]  == 255:
					mask_actual[i][j] = 1



		accuracy_mat = np.logical_xor(F, mask_actual) 
	 
		accuracy = np.sum(accuracy_mat)
		print("Accuracy: ", 100 - accuracy/(h*w), "%")

	sp_lab=np.zeros(I.shape,dtype=np.uint8)
	for sp in SP_list:
		for pixels in sp.pixels:
			i,j=pixels
			sp_lab[i][j]=sp.mean_lab
	sp_lab=cv2.cvtColor(sp_lab, cv2.COLOR_Lab2RGB)
	
	plt.subplot(2,2,1)
	plt.tick_params(labelcolor='black', top='off', bottom='off', left='off', right='off')
	plt.imshow(I[...,::-1])
	plt.axis("off")
	plt.xlabel("Input image")

	plt.subplot(2,2,2)
	plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
	plt.imshow(I_marked[...,::-1])
	plt.axis("off")
	plt.xlabel("Super-pixel boundaries and centroid")

	plt.subplot(2,2,3)
	plt.imshow(sp_lab)
	plt.axis("off")
	plt.xlabel("Super-pixel representation")


	plt.subplot(2,2,4)
	plt.imshow(Final[...,::-1])
	plt.axis("off")
	plt.xlabel("Output Image")


	
	cv2.imwrite("out.png",Final)
	plt.show()

if __name__ == '__main__':
	main()
