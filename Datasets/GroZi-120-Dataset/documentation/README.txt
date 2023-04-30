GROZI DATABASE

Created Summer 2006 - University of California, San Diego

120 Grocery Products Categories, collected from the web and from the Sunshine Store(R) @ UCSD 

-----------------------------------------------------------------------------------------------------------

Organization of the folders:

- Index : contains - 120 jpeg images, one per product, indexed from 1.jpeg to 120.jpeg

                   - UPC_index.txt : List of products in the database, with UPC code.
			   
			   Format:  [Index of product in the database] (from 1 to 120) 
                              [UPC code] 
                   		[Product description]   	   

- Video : contains 29 Divx movie files, named Shelf_1.avi, ... , Shelf_29.avi

- inVitro: Folders 1 to 120 (1 folder per product). Each folder contains : 
   
       - info.txt  
		
	   Format : [web] : [number of images obtained from the web] (stored in subfolder web/JPEG)
				
		      [video] : [number of images cropped from the videos] (stored in subfolder video)	

                  [x1] [Shelf_y1] -> x1 samples from video Shelf_y1
                  [x2] [Shelf_y2] -> x2 samples from video Shelf_y2	
			 .       .         

	- Folder web : 
      
             - Folder JPEG : contains the .jpg images originally downloaded from the web. They are named web%d.jpg
		
	       - Folder PNG : contains the images from the web, processed with Photoshop. The background was manually set to transparent. They are named web%d.png

		 - Folder masks : contains binary masks to separate the product from the background. They are named mask%d.png

               

- inSitu: Folders 1 to 120 (1 folder per product). Each folder contains : 
   
       - info.txt  
		
	   Format : [web] : [number of images obtained from the web] (stored in subfolder web/JPEG)
				
		      [video] : [number of images cropped from the videos] (stored in subfolder video)	

                  [x1] [Shelf_y1] -> x1 samples from video Shelf_y1
                  [x2] [Shelf_y2] -> x2 samples from video Shelf_y2	
			 .       .         
			 .       .
			 .       .
	 
	 - coordinates.txt : contains the coordinates of the boxes (stored in subfolder video) with the product cropped form the videos

	   Format : [a] [frame] [x toplfet corner] [y toplfet corner] [width] [height] -> box in video Shelf_a.avi
			[a] [frame] [x toplfet corner] [y toplfet corner] [width] [height] -> box in video Shelf_a.avi
			[b] [frame] [x toplfet corner] [y toplfet corner] [width] [height] -> box in video Shelf_b.avi
			[b] [frame] [x toplfet corner] [y toplfet corner] [width] [height] -> box in video Shelf_b.avi
			 .     .             .                 .              .       .
			 .     .             .                 .              .       .
			 .     .             .                 .              .       .


      			|---------------------> x
	       		|
		  		|
				|       
				|   tlc.______
				|      |      | height
				|      |______| 
				|        width
				|
				|
				|
				v  y
           
     
	- Folder video : contains the .png images of the product, cropped fom the videos. They are called video%d.png
           