I, mode, I_dummy

%input image and scribbling

[h,w,c] = size(I);
region_size=20;


% I_lab=cv2.cvtColor(I, cv2.COLOR_BGR2Lab)

I_lab = rgb2lab(I); %Check
[SP_labels , Num_labels] = gen_sp_slic(I,region_size);

SP_list_bools=zeros(Num_labels , 1);
SP_list = [];
for i = 1:Num_labels    
    s = SPNode;
    SP_list = SP_list + [s];
end

for i =1:h
    for j = 1:w
        if SP_list_bools(i,j) == 0
            SP_list_bools(i,j)=1;
            SP_list(SP_labels(i,j)).label = SP_label(i,j);
            SP_list(SP_labels(i,j)).pixels = SP_list(SP_labels(i,j)).pixels + [[i,j]];
        else
            SP_list(SP_labels(i,j)).pixels = SP_list(SP_labels(i,j)).pixels + [[i,j]];            
        end
    end
end

for idx1 = 1:Num_labels
    sp = SP_list(idx1);
    n_pixels = length(sp.pixels)
    i_sum = 0;
    j_sum = 0;
    lab_sum = [0,0,0];
    tmp_mask = zeros(h,w,'unit8');
    for idx2 = 1:n_pixels
        each = sp.pixels(idx2);
        [i,j] = each;
        i_sum = i_sum + i;
        j_sum = j_sum + j ;
        lab_sum = lab_sum + I_lab(i,j);
        tmp_mask(i,j) = 255;
    end
    % sp.lab_hist=cv2.calcHist([I_lab],[0,1,2],tmp_mask,lab_bins,l_range+a_range+b_range)
    sp.centroid = sp.centroid + [[idivide(i_sum,n_pixels), idivide(j_sum,n_pixels)]]
    sp.mean_lab = lab_sum / n_pixels;
    sp.real_lab = [sp.mean_lab(1)*100/255,sp.mean_lab(2)-128,sp.mean_lab(3)-128];
end

for idx = 1:length(marked_ob_pixels)
    [x,y] = marked_ob_pixels(idx);
    SP_list(SP_labels(x,y)).type = 'ob';

end

for idx = 1:length(marked_bg_pixels)
    [x,y] = marked_bg_pixels(idx);
    SP_list(SP_labels(x,y)).type = 'bg';
    
end
I_marked = draw_sp_mask(I,SP_labels);


mask_ob  = zeros(h,w , 'uint8');
mask_bg  = zeros(h,w , 'uint8');


for idx = 1:length(marked_ob_pixels)
    [x,y] = marked_ob_pixels(idx);
    mask_ob(x,y) = 255;
end
for idx = 1:length(marked_bg_pixels)
    [x,y] = marked_bg_pixels(idx);
    mask_bg(x,y) = 255;

% calcHist();
%

G = gen_graph(I_lab , SP_list , hist_ob , hist_bg);
for i  = 1:Num_labels
    each  = G.nodes(i)



function [SP_Labels,NumLabels] = get_sp_slic(I , region_size)
    [SP_Labels,NumLabels] = superpixels(I,region_size , 'IsInputLab' , true , 'NumIterations' , 20);
end

function dist = distance(p0,p1)
    dist = sqrt((p0(0) - p1(0))^2 + (p0(1)-p1(1))^2);
end

function dist_3d = distance_3d(p0,p1)
    dist_3d = sqrt(p0(0) - p1(0)^2 + (p0(1)-p1(1))^2 + (p0(2) - p1(2))^2);
end

function sp_mask = draw_sp_mask(A, L)
    BW = boundarymask(L);    
    sp_mask = imoverlay(A,BW,[0.5 0.5 0.5]);
end

function G = gen_graph(I, SP_list, hist_ob, hist_bg)
    G = graph();
    s = SPNode;
    s.label='s';
    t = SPNode;
    t.label='t';
    lambda_=.9
	sig_=5
	%----------------------------------------
    hist_ob_sum=int64(hist_ob.sum());  %yeh dekhna hai
	hist_bg_sum=int64(hist_bg.sum())
    %----------------------------------------

    
    for idx = 1:len(SP_list)
        u = SP_list(i);
        K=0;
		region_rad = sqrt(len(u.pixels)/pi)

		for idx2 = 1:len(SP_list)
            v = SP_list(idx2);

			if u ~= v:
				if distance(u.centroid, v.centroid) <= 2.5*region_rad:
					sim = math.exp(-(cv2.compareHist(u.lab_hist,v.lab_hist,3)**2/2*sig_**2))*(1/distance(u.centroid, v.centroid))
					K+=sim
					G.add_edge(u, v, sim=sim)


        %to be done
    end    


