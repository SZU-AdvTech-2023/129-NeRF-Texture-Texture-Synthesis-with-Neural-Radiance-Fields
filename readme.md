<h1 align="center">
  NeRF-Texture: Texture Synthesis with Neural Radiance Fields
</h1>

*Given a set of multi-view images of the target texture with meso-structure, our model synthesizes Neural Radiance Field (NeRF) textures, which can then be applied to novel shapes, such as the skirt and hat in the figure, with rich geometric and appearance details.*

***

# Environment

**1.Begin by setting up a Python 3.7+ environment with [PyMesh](https://pymesh.readthedocs.io/en/latest/installation.html) and [frnn](https://github.com/lxxue/FRNN).**

**2.Install [PyTorch](https://pytorch.org/), [PyTorch3D](https://pytorch3d.org/), [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), and [cubvh](https://github.com/ashawkey/cubvh) by invoking:**
(if failed please refer to the official webs)

    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    pip install git+https://github.com/ashawkey/cubvh

**3.Install [raymarching](./raymarching/)**

    cd raymarching
    python setup.py build_ext --inplace
    pip install .
    cd ..

**4.Install [gridencoder](./gridencoder/)**

    cd gridencoder
    python setup.py build_ext --inplace
    pip install .
    cd ..

**5.Install [RayTracer](./external/RayTracer/) by invoking:**
(modified from [raytracing](https://github.com/ashawkey/raytracing))

    cd external/RayTracer
    python setup.py develop
    cd ../..

**6.Then invoke:**

    pip install -r requirements.txt

**7.Download and compile [CoACD](https://github.com/SarahWeiii/CoACD) and [Manifold](https://github.com/hjwdzh/Manifold):**

The detailed instructions for installation refer to official websites. After compiling both CoACD and Manifold, rename the binary file 'main' of [CoACD](https://github.com/SarahWeiii/CoACD) as 'CoACD' and move it to ./tools. Also move binary files 'manifold' and 'simplify' of [Manifold](https://github.com/hjwdzh/Manifold) to ./tools.

Please refer to the respective official websites for installation details.
Once you've successfully compiled both CoACD and Manifold, make the following changes:

(1) Rename the binary file 'main' from the CoACD compilation to 'CoACD'.

(2) Move this renamed 'CoACD' file into the ./tools directory.

(3) Similarly, relocate the 'manifold' and 'simplify' binary files, which result from the Manifold compilation, to the ./tools directory.

By following these instructions, you will successfully set up the necessary binaries in the correct location.

# Prepare Your Data

A dataset is a directory with the following structure:

    ${DATA_NAME}
        ├── ${DATA_NAME}.mp4
        ├── images
        │   └── ${id}.png
        ├── transforms.json

At a high level, a dataset consists of the following:
 * A captured video of the target texture.
 * A collection of images from the video.
 * Camera parameters for each image.

We provide scripts for users to preprocess their own captured data. Please follow the steps below:

## 1. Data capture

(1) Capture a multi-view video of the taret texture and name it as ${DATA_NAME}.

(2) Create a directory named ${DATA_NAME} and move the video to it. The following structure should obtained:

    ${DATA_NAME}
        ├── ${DATA_NAME}.mp4

## 2. Download [MiVOS](https://github.com/hkchengrex/MiVOS)
Download [MiVOS](https://github.com/hkchengrex/MiVOS). In the file [prepare_data.py], rename the variable 'MIVOS_PATH’ as the path to the MiVOS installation.

## 3. Run script
Before beginning the data preprocessing, it's crucial to assign values to the 'path_to_dataset' and 'dataset_name' variables within the prepare_your_data.py script. These variables define the directory path to your dataset and the dataset name, respectively.

Once these variables are set, you can run the [prepare_data.py] script to commence data preprocessing. This process includes several steps such as frame extraction, blurred image removal, image segmentation, and camera pose estimation.

    python prepare_data.py

During the image segmentation phase, you will need to interact with the script by clicking on regions containing textures or objects within the video to effectively segment it.

Upon successful execution of the script, the data will be organized into the necessary directory structure for the next steps of the process.

## 4. Interactive Segmentation

During the execution of the script, an interactive segmentation window will appear. Follow the instructions below to perform the interactive segmentation:

(1) Left-click on the regions you want to segment. This will mark those regions for segmentation.

(2) Right-click on the regions you do not want to keep. This will exclude those regions from segmentation.

(3) Click on the "Propagate" button to propagate the segmentation result to all frames. This ensures consistent segmentation across the entire video.

(4) Click on the "Play" button to check the video segmentation. This will start playing the video with the applied segmentation.

(5) Click on the "Play" button again to stop the video playback.

(6) Click on the "Save" button to save the segmentation results. This will save the segmented frames and associated metadata.

(7) Close the segmentation window when the terminal shows that the results saving is done. The script will continue its execution.

By following these steps, you can interactively segment the video and save the segmentation results for further processing.


# Training

## Coarse shape estimation
After preparing a dataset, you can train a NeRF-Texture based on it. But before that, you need to train an Instant-NGP to estimate the coarse shape by running:

    CUDA_VISIBLE_DEVICES=0 python main_nerf.py ${PATH_TO_DATASET} --workspace ./logs/${DATA_NAME} -O --bound 1.0 --scale 0.8 --dt_gamma 0 --ff --mode colmap --gui

Please follow these instructions to navigate through the interactive window:

**1. Train**: Upon opening the interactive window, locate the 'start' button to commence the training of Instant-NGP. The process should reach convergence in approximately one minute. Once it does, you can halt the training by clicking on the 'stop' button, which is located in the same area as the 'start' button.

**2. Obtain Raw Base Mesh** : Next, click on the 'save_mesh' button. This action will generate the raw base mesh using the Marching Cubes algorithm on the density field.

**3. Extract Scan Point Cloud** : Proceed to click on the 'save_pcl' button. This will extract the scan point cloud, which will subsequently be utilized to determine the height threshold of the meso-structure texture.

**4. Close the Window**: Finally, upon completion of all the above steps, you can close the interactive window.


## Train NeRF-Texture

Modify the PATH_TO_DATASET and DATA_NAME defined in [data_args.py], then run [main.py].

    CUDA_VISIBLE_DEVICES=0 python main.py

Details pertaining to model training and texture synthesis will be elaborated in the upcoming sections of this content

# Texture synthesis

The summary of texture synthesis steps is illustrated as the following steps:

## 1. Start training NeRF-Texture

To initiate model training, please click on the 'start' button. Once the training converges, press the 'stop' button, which is located at the same place. It is recommended that the training time falls within the 10 to 20-minute range for optimal results.

## 2. Save checkpoint

Click on the 'save' button to save checkpoints of the trained model.

## 3. Create picked_faces.obj (optional)

The implicit patches are, by default, sampled on the base mesh, which is labeled as 'surface_coacd_remesh.obj' and located in the 'meshes' folder under the logs directory. However, there might be occasions when you wish to sample textures from specific regions instead of the entire base mesh. This could be due to certain areas not being adequately reconstructed, or because the desired textures are found within certain regions.

To customize the regions for texture sampling, you can create a 'picked_faces.obj' file and place it in the 'meshes' folder, the same location where the base mesh is stored.

Here are the steps to achieve this:

(1) Duplicate the base mesh file in the 'meshes' folder and rename the copy as 'picked_faces.obj'.

(2) Next, use mesh edit tools (e.g. MeshLab) to remove the regions that are not needed for texture sampling, keeping only the areas you wish to sample texture patches from.

This way, you can control where textures are sampled from on your mesh.

## 4. Sample implicit patches

To sample implicit patches, click on the 'sample patches' button. This action will store the sampled patches in a '.npz' file within the 'field' folder, located under the logs directory. The progress will be displayed in the terminal. 

Please note that the colors of the rendered patches will also be saved for verification purposes.
If the size of the patches (not the resolution) appears too large or too small, you can adjust the value of the **pattern_rate** parameter. This parameter is defined in the [data_args.py] file and serves as the scale factor of the pixel gap, relative to the average edge length of the base mesh. Adjusting the pattern_rate accordingly will allow you to achieve the desired patch size.

## 5. Synthesize implicit texture

You will need to assign values to the 'DATA_NAME' and 'MODEL_NAME' variables in the [matching_and_quilting.py], and then proceed to execute the code. By default, the 'MODEL_NAME' is set to 'curved_grid_hash_clus_optcam_SH'. If you are unsure about this, please refer to the 'field' folder located within the log directory for confirmation. The synthesized result will be saved in a **'texture.npz'** file in the 'field' folder under the log directory.

The following image provides a visual illustration of the processes of patch sampling, matching, and quilting. The resulting synthesized implicit texture encapsulates both the meso-scale geometry and the view-dependent appearance, which can be subsequently decoded by a NeRF (Neural Radiance Fields) network.

## 6. Load synthesized implicit textures

Click on the 'load synthesis' button to load the synthesized texture stored in the **'texture.npz'** file. Then the synthesized texture will then be dislpayed within the interactive window.

## 7. Apply synthesized textures to shapes

To apply the synthesized texture to a shape, first place a 'target.obj' file into the 'field' folder. After that, click on the 'load_shape' button to apply the texture. Please remember that you must first perform the 'load synthesis' operation before proceeding to 'load shape'.



