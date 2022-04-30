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
algo = args.algo
sp_en = args.sp_en
acc = args.acc

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



class SPNode():

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
	h, w, _ = I_dummy.shape

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
	SLICO = 101
	num_iter = 4
	sp_slic = cv2.ximgproc.createSuperpixelSLIC(
	    I, algorithm=SLICO, region_size=region_size_, ruler=10.0)
	sp_slic.iterate(num_iterations=num_iter)

	return sp_slic


def draw_centroids(I, SP, SP_list):
	mask = SP.getLabelContourMask()
	I[ mask == -1 ] = [128,128,128]
	I[ mask == 255] = [128,128,128]
	for each in SP_list:  
		I[each.centroid] = 128
	return I


def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


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

			l_i = l_//((l_range[1]-l_range[0])//lab_bins[0])
			a_i = a_//((a_range[1]-a_range[0])//lab_bins[1])
			b_i = b_//((b_range[1]-b_range[0])//lab_bins[2])
			pr_ob = hist_ob[l_i, a_i, b_i]/hist_ob_sum
			pr_bg = hist_bg[l_i, a_i, b_i]/hist_bg_sum
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
	inputfile = args.img
	print('Using image: ', inputfile)

	I = cv2.imread('./images/input/'+inputfile)  # imread wont rise exceptions by default
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

	I_lab = np.array(cv2.cvtColor(I, cv2.COLOR_BGR2Lab))
	# 	I_lab = cv2.cvtColor(I, cv2.COLOR_BGR2Lab)
	SP = gen_sp_slic(I, region_size)
	SP_labels = SP.getLabels()
	SP_list = [None]*SP.getNumberOfSuperpixels()

	for i in range(h):
		for j in range(w):
			if not SP_list[SP_labels[i][j]]:
				SP_list[SP_labels[i][j]] = SPNode()
				SP_list[SP_labels[i][j]].label = SP_labels[i][j]

			SP_list[SP_labels[i][j]].pixels.append((i, j))
	SP_list = list(filter(None, SP_list))
	for sp in SP_list:
			n_pixels = len(sp.pixels)
			i_sum,j_sum = np.sum(sp.pixels , (0))
			lab_sum = np.zeros(3)
			tmp_mask = np.zeros((h, w), np.uint8)
			for each in sp.pixels:
				lab_sum = lab_sum + I_lab[each]
				tmp_mask[each] = 255
			sp.lab_hist = cv2.calcHist([I_lab], [0, 1, 2], tmp_mask, lab_bins, l_range+a_range+b_range)
			sp.centroid += (i_sum//n_pixels, j_sum//n_pixels,)
			sp.mean_lab = lab_sum / n_pixels
			sp.real_lab = [sp.mean_lab[0]*100/255,
			    sp.mean_lab[1]-128, sp.mean_lab[2]-128]
	mask_ob = np.zeros((h, w), dtype=np.uint8)
	mask_bg = np.zeros((h, w), dtype=np.uint8)

	for pixels in marked_ob_pixels:
		SP_list[SP_labels[pixels]].type = "ob"
		mask_ob[pixels] = 255
	for pixels in marked_bg_pixels:
		SP_list[SP_labels[pixels]].type = "bg"
		mask_bg[pixels] = 255

	I_marked = draw_centroids(I,SP, SP_list)


	hist_ob = cv2.calcHist([I_lab], [0, 1, 2], mask_ob,
	                       lab_bins, l_range+a_range+b_range)

	hist_bg = cv2.calcHist([I_lab], [0, 1, 2], mask_bg,
	                       lab_bins, l_range+a_range+b_range)

	G = gen_graph(I_lab, SP_list, hist_ob, hist_bg)

	for each in G.nodes():
		if each.label == 's':
			s = each
		if each.label == 't':
			t = each
	# global algo
	if algo == "bk":
		RG = boykov_kolmogorov.boykov_kolmogorov(G, s, t, capacity='sim')
		source_tree, _ = RG.graph['trees']
		source_label_list = set(source_tree)
	else:
		source_label_list = ff.ford_fulkerson(G, s, t)

	F = np.zeros((h, w), dtype=np.uint8)
 
	for sp in source_label_list:
		for pixels in sp.pixels:
			F[pixels] = 1
	Final = cv2.bitwise_and(I, I, mask=F)
	print("------------------\n","Time taken : ",time.process_time() - start,"\n------------------")
	global acc 
	if(acc=="y"):
		mask_actual_img = cv2.imread('./images/masks/'+inputfile.split('.')[0]+'_mask.png')
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
			sp_lab[pixels]=sp.mean_lab
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


	
	cv2.imwrite("./images/output/"+inputfile.split('.')[0]+"_output.png",Final)
	plt.show()

if __name__ == '__main__':
	main()
