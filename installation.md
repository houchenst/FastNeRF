#Getting Set Up

Setting up the conda environment and downloading data and weights for the fern scene
```
conda env create -f environment.yml
conda activate nerf
bash download_example_data.sh
```
Double check that within the 'logs' directory there is a folder titled 'fern_example'
Within 'fern_example' should be 'config.txt', 'args.txt', and two .npy files of model weights

#Rendering!

You can try rendering a low resolution image using render_demo.py

To render with normal nerf:
```
python render_demo.py --vanilla
```

To render using point cloud depth estimation:
```
python render_demo.py --cloud
```

To render using sparse sampling + interpolation:
```
python render_demo.py --interpolate
```

The using --cloud may lead to slow rendering.
You may also pass in an integer using the --down flag. Both dimensions of the new views are divided by this number. The default is 12 and it is not recommended to go lower that 4 if you aren't using a GPU. 
NOTE: If you are passing --interpolate, then --down must be a product of elements of the set {2,2,3,3,7} ¯\_(ツ)_/¯ it's a feature!

To visualize the depth bounds and the point cloud
```
python render_demo.py --cloud_viz
```




