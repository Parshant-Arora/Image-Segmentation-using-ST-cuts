## Project Description:

This is the course project for Medical Image Computing (CS736) wheren we explored different methods for performing image segmentation using s-t cuts. Essentially imgae segmentation using s-t cuts involves the use of max-flow min-cut algoithms to segment a given image into two segments pertaining to foreground or background. The general algorithm uses a graph modelled on the image and pixels as it is. We modify this algorithm by introducing superpixels on the graph that essentially clusters various pixels that share similar charateristics. This is followed by use of a different more efficient graph-cut algorithm namely Boykov-Kolmogorov algorithm. 

We compare the results of these with various cases where we use Boykov with and without super-pixelization and compare the results with Ford-dulkerson with and without sp. The metrics on which we compare the results are time taken and accuracy. Accuracy is measured on the basis of difference between a manually created mask and the algo generated mask. 


## Dependencies:

All dependencies are enlisted in requirements.txt
	
	Install them using : `pip install -r requirements.txt`


## Executing the code:

1. Run the main file using python3: `python3 fast_seg.py -i <input-image>`
	* Will provide a minimal GUI to mark the seed pixels. While marking, switching between "background" and "object" pixels are done using keys 'b' and 'o' respectively. By default GUI initializes in object mode. Object is marked with "red" markings and Background with "blue".
	* Use `python3 fast_seg.py -h` for help
2. Press ESC after marking the seeds.
3. Output window will provide the results.
4. Output image will be written in running folder, named "out.png"



## Research paper:
Research paper that we referred can be downloaded from [here](https://www.ijitee.org/wp-content/uploads/papers/v8i8/H7423068819.pdf).
