# INF574 - PointNet++ implementation

This project was done by Adrien Ramanana Rahary and Virgile Foussereau during a course on Shape Analysis given by Mathieu Desbrun.

## Installation

The dataset being too large, it is not provided in the submission. Please install ShapeNetPart from [this repo](https://github.com/AnTao97/PointCloudDatasets) in the `data/PointCloudDatasets` folder of the project. To do so, open the folder:

```bash
cd data/PointCloudDatasets
```

Then paste the folder `shapenetpart_hdf5_2048` directly downloaded from this [link](https://cloud.tsinghua.edu.cn/f/c25d94e163454196a26b/).

## Required packages

Make sure to have at least matplotlib and pytorch installed. A list of packages from our environment and their versions is available in `environment.yml`.

## Usage

The project contains 2 already trained models: `model_airplane.pt` which has been trained using our implementation of PointNet++ and `model_custom_airplane.pt` which has been trained using our slightly modified version of PointNet++. By default, `model_custom_airplane.pt` is used when you run the file, but you can change it in the code to test the other model as well.

From the main `project_INF554` folder, run this command: 
```bash
python3 PointNetpp.py
```

By default, it will show the result of our model on a test example using matplotlib. If you want to retrain a model, you can open the file and uncomment `train()` at the very end of the file. The newly trained model will be saved as `model.pt`. If you want to evaluate a model accuracy, you can uncomment `eval()` at the very end of the file.