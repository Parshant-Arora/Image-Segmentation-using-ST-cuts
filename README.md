## Project Description:

This is the course project for Medical Image Computing (CS736) wheren we explored different methods for performing image segmentation using s-t cuts. Essentially imgae segmentation using s-t cuts involves the use of max-flow min-cut algoithms to segment a given image into two segments pertaining to foreground or background. The general algorithm uses a graph modelled on the image and pixels as it is. We modify this algorithm by introducing superpixels on the graph that essentially clusters various pixels that share similar charateristics. This is followed by use of a different more efficient graph-cut algorithm namely Boykov-Kolmogorov algorithm. 

We compare the results of these with various cases where we use Boykov with and without super-pixelization and compare the results with Ford-dulkerson with and without sp. The metrics on which we compare the results are time taken and accuracy. Accuracy is measured on the basis of difference between a manually created mask and the algo generated mask. 


## Dependencies:

All dependencies are enlisted in requirements.txt
	
	Install them using : `pip install -r requirements.txt`


## Executing the code:

**Naming convention**

	1. **image.png** - to be segmented
	2. **image_mask.png** - mask, which is used for determining accuracy of the result obtained 
	3. **image_actual.png** - mask applied on given image (not needed for code though) 
	4. **image_seg.png** - output image
	

**Note:**

	● Extension need not be png
	
**Runtime Commands**

	**fast_seg.py**
	The main code used to produce the segmentation results. command-line arguments
	**-i / --img** : -i <path to input image>
	**-a / --algo** : values “bk”/”ff”
	**“bk”** - used to perform segmentation using boykov kolmogorov algorithm
	**“ff”** - used to perform segmentation using ford fulkerson algorithm -s/ --sp_en : values“y”/”n”
	**“y”** - used to enable superpixalization
	**“n”** - used to disable superpixalization -ac / --acc : values“y”/”n”
	**“y”** - used to enable accuracy metric “n” - used to disable accuracy metric

	**Example:** python3 fast_seg.py -i bunny.png -a bk -s y -ac y

**Note:**

	● To use the accuracy metric, mask used for measuring accuracy should be present in the same path folder as the image needed to be segmented.
	● All the arguments mentioned above are required.
	● Install all the dependencies before running the program (refer to requirements.txt)


**Other files**

	1. **boykov_kolmogorov.py** : used for obtaining min cut using boykov kolmogorov algorithm 
	2. **ff.py** : used for obtaining min cut using ford fulkerson algorithm
 	3. **Superpixels.py** : scratch code to generate super pixels (not used in fast_seg.py) 
	4. **create_mask.py** : used for creating manual mask (not used in fast_seg.py)


## Results 

The segmented images:
<img width="851" alt="Screenshot 2022-04-30 at 3 16 03 PM" src="https://user-images.githubusercontent.com/81502104/166100587-dd14d19f-c660-4531-8c05-f84e299d3f16.png">

Comparison:
<img width="938" alt="Screenshot 2022-04-30 at 3 17 31 PM" src="https://user-images.githubusercontent.com/81502104/166100609-d21ace06-6e7b-41a9-adc1-300fbc381fcd.png">


## Research paper:
Research paper that we referred can be downloaded from [here](https://www.ijitee.org/wp-content/uploads/papers/v8i8/H7423068819.pdf).
