import torch
import torch.nn as nn
import numpy as np
import sys
import os
root= os.getcwd()

sys.path.append(os.path.join(root, 'data/PointCloudDatasets'))
import dataset
import matplotlib.pyplot as plt


rootDataset = os.path.join(root, 'data/PointCloudDatasets')

class PointSet:
    def __init__(self, coordinates, features):
        self.coordinates = coordinates
        self.features = features

    def __getitem__(self, item):
        return self.coordinates[item], self.features[item]

    def __len__(self):
        return len(self.coordinates)

    def __str__(self):
        return 'PointSet: {} points, {} features'.format(len(self.coordinates), len(self.features[0]))

    

def sampling_layer(pointSet, num_points=1024):
    #compute the distances 
    points = pointSet.coordinates.detach().numpy()
    remaining_points = np.array([i for i in range(len(points))])
    sampled_points = np.array([0]*num_points)
    distances = np.ones_like(remaining_points) * np.inf
    selected = np.random.choice(remaining_points)
    sampled_points[0] = selected
    remaining_points = np.delete(remaining_points, selected)
    for i in range(1, num_points):
        dist = np.linalg.norm(points[remaining_points] - points[selected], axis=1)
        distances[remaining_points] = np.minimum(distances[remaining_points], dist)
        ## select the point with the maximum distance
        selected = np.argmax(distances[remaining_points])
        sampled_points[i] = remaining_points[selected]
        remaining_points = np.delete(remaining_points, selected)
    return PointSet(pointSet.coordinates[sampled_points], pointSet.features[sampled_points])

def custom_sampling_layer(pointSet, num_points=1024):
    points = pointSet.coordinates.detach().numpy()
    remaining_points = np.array([i for i in range(len(points))])
    sampled_points = np.array([0]*num_points)
    distances = np.ones_like(remaining_points) * np.inf
    selected = np.random.choice(remaining_points)
    sampled_points[0] = selected
    remaining_points = np.delete(remaining_points, selected)
    for i in range(1, num_points):
        dist = np.linalg.norm(points[remaining_points] - points[selected], axis=1)
        distances[remaining_points] = np.minimum(distances[remaining_points], dist)
        ## select the point with probability proportional to the squared distance
        probabilities = distances[remaining_points]**2 / np.sum(distances[remaining_points]**2)
        selected = np.random.choice(range(len(remaining_points)), p=probabilities)
        sampled_points[i] = remaining_points[selected]
        remaining_points = np.delete(remaining_points, selected)
    return PointSet(pointSet.coordinates[sampled_points], pointSet.features[sampled_points])



def ball_query(query_point_coordinates, pointset, K, R):
    """
    - query_point_coordinates is the point we are trying to find the neighbours of
    - pointset is the set of points we are finding the neighbours in
    - K is the maximum number of neighbours we are returning
    - R is the radius of the ball
    """
    coords = pointset.coordinates.detach().numpy()
    dist_to_query_point = np.linalg.norm(coords - query_point_coordinates, axis = 1)

    return PointSet(pointset.coordinates[dist_to_query_point <= R][:K], pointset.features[dist_to_query_point <= R][:K])


def grouping_layer(pointset, centroids, K, r=0.2):
    """
    The input to this layer is:
    - pointset: a point set of size N × (d + C)
    - centroids: the coordinates of a set of centroids of size N0 × d.

    The output are groups of point sets of size N0 × K × (d + C).
    Each group corresponds to a local region and K is the number of points in the neighborhood of centroid points.
    """
    # 0.2 is the radius with which the authors of the original paper got the best results
    centroids = centroids.detach().numpy()

    result = [ball_query(centroid, pointset, K, r) for centroid in centroids]
    return result

## PointNet layer

class PointNetLayer(nn.Module):
    def __init__(self, in_channels, layers, globalLayer=False):
        super(PointNetLayer, self).__init__()
        self.fc_layers = []
        for i in range(len(layers)):
            if globalLayer:
                if i == 0:
                    fc = nn.Sequential(
                        nn.Linear(in_channels, layers[i]),
                        nn.ReLU()
                    )
                else:
                    fc = nn.Sequential(
                        nn.Linear(layers[i-1], layers[i]),
                        nn.ReLU()
                    )
            else:
                if i == 0:
                    fc = nn.Sequential(
                        nn.Linear(in_channels, layers[i]),
                        nn.ReLU()
                    )
                else:
                    fc = nn.Sequential(
                        nn.Linear(layers[i-1], layers[i]),
                        nn.ReLU()
                    )
            self.fc_layers.append(fc)
        self.fc_layers = nn.ModuleList(self.fc_layers)
        

    def forward(self, x):
        for fc in self.fc_layers:
            x = fc(x)
        x = torch.max(x, dim=0, keepdim=True)[0]
        return x

class UnitPointNet(nn.Module):
    def __init__(self, in_channels, layers, last):
        super(UnitPointNet, self).__init__()
        self.fc_layers = []
        for i in range(len(layers)):
            if i == 0:
                fc = nn.Sequential(
                    nn.Linear(in_channels, layers[i]),
                    nn.BatchNorm1d(layers[i]),
                    nn.ReLU()
                )
            elif last and i == len(layers) - 1:
                fc = nn.Sequential(
                    nn.Linear(layers[i-1], layers[i]),
                )
            else:
                fc = nn.Sequential(
                    nn.Linear(layers[i-1], layers[i]),
                    nn.BatchNorm1d(layers[i]),
                    nn.ReLU()
                )
            self.fc_layers.append(fc)
        if not last:
            self.fc_layers.append(nn.Dropout(0.5))
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, x):
        for fc in self.fc_layers:
            x = fc(x)
        return x


class SetAbstraction(nn.Module):
    def __init__(self, K, r, layers, C):
        super(SetAbstraction, self).__init__()
        self.K = K
        self.r = r
        in_channels = C + 3
        self.pointnet = PointNetLayer(in_channels, layers, globalLayer=False)
    
    def transform(self, regions, centroids):
        for i in range(len(regions)):
            regions[i].coordinates = regions[i].coordinates - centroids[i]
        return regions

    def abstract(self, pointset):
        centroids = custom_sampling_layer(pointset, self.K)
        regions = grouping_layer(pointset, centroids.coordinates, self.K, self.r)
        regions = self.transform(regions, centroids.coordinates)
        groups = []
        for region in regions:
            groups.append(torch.cat((region.coordinates, region.features), dim = 1))
        for i in range(len(groups)):
            groups[i] = self.pointnet(groups[i])
        new_pointset = PointSet(centroids.coordinates, torch.cat(groups, dim = 0))
        return new_pointset

class GlobalSetAbstraction(nn.Module):
    def __init__(self, layers, C):
        super(GlobalSetAbstraction, self).__init__()
        in_channels = C + 3
        self.pointnet = PointNetLayer(in_channels, layers, globalLayer=True)

    def abstract(self, pointset):
        x = torch.cat((pointset.coordinates, pointset.features), dim = 1)
        x = self.pointnet(x)
        new_pointset = PointSet(torch.zeros((1, 3)), x)
        return new_pointset

def point_feature_propagation(p, k, propagated_ptcloud, interpolated_ptcloud):
    """Returns ONLY the interpolated features. interpolated_features[i] corresponds to the features of point i in interpolated_ptcloud"""

    coords = propagated_ptcloud.coordinates.detach().numpy()
    feats = propagated_ptcloud.features.detach().numpy()
    interpolated_coords = interpolated_ptcloud.detach().numpy()
    N_l = np.shape(coords)[0]
    N_lminusone = np.shape(interpolated_coords)[0]

    k = min(N_l, k) #We cannot take more neighbours in the computation than there are elements in the propagated pointcloud

    #Compute the distances and sort them to get the k nearest neighbours
    dist = np.linalg.norm(coords - interpolated_coords[:,None,:], axis = 2)
    sorted_idx = np.argsort(dist, axis = 1)
    sorted_dist = np.sort(dist, axis = 1)

    #Problem: If a point belongs to the 2 sets, distance will be 0 and we won't be able to take the inverse squared
    #Solution: We threshold distances so that they do not go below a certain threshold
    threshold = 0.000001
    sorted_dist = np.where(sorted_dist < threshold, threshold, sorted_dist)

    #Compute the weights using the k nearest neigbours
    w = sorted_dist[:,:k] **(-p)
    w_sum = np.sum(w, axis = 1)

    #Prepare the array that will contain the interpolated features
    C = np.shape(feats)[0]
    feature_size = np.shape(feats)[1]

        #Create a matrix which contrains all the features and that can be sorted easily
    propagated_ptcloud_features = np.transpose(np.repeat(feats, N_lminusone, axis = 0).reshape(N_l,N_lminusone,feature_size), (1,0,2))

        #Sort the matrix to keep the features of the k nearest neighbours of i on line i of the matrix
    selected_sorted_features = np.array(list(map(lambda x, y: y[x], sorted_idx[:,:k], propagated_ptcloud_features)))

        #Rearrange the weights so that we can easily multiply the features by the weights
    tiled_w = np.tile(w[:,None,:], (feature_size,1))
    transposed_tiled_w = np.transpose(tiled_w,(0,2,1))
    weighted_features = selected_sorted_features * transposed_tiled_w
    weighted_features = np.sum(selected_sorted_features * transposed_tiled_w, axis = 1)
    #print(weighted_features)

        #Divide the weighted features by the sum of weights by doing the same trick as 2 lines above
    #print("---")
    tiled_wsum = np.tile(w_sum[None,:], (feature_size,1))
    transposed_tiled_wsum = tiled_wsum.T
    #print(transposed_tiled_wsum)
    interpolated_features = weighted_features / transposed_tiled_wsum

    result = torch.from_numpy(interpolated_features).float()

    return result




class FeaturePropagation(nn.Module):
    def __init__(self, layers, C, last):
        super(FeaturePropagation, self).__init__()
        in_channels = C + 3
        self.unitPointNet = UnitPointNet(in_channels, layers, last)
    
    def forward(self, pointset, original_pointset):
        p = 2
        k = 3
        new_pointset = PointSet(original_pointset.coordinates, original_pointset.features)
        interpolated_features = point_feature_propagation(p,k, pointset, original_pointset.coordinates)
        new_pointset.features = torch.cat((new_pointset.features, interpolated_features), dim = 1)
        x = torch.cat((new_pointset.coordinates, new_pointset.features), dim = 1)
        x = self.unitPointNet(x)
        new_pointset = PointSet(new_pointset.coordinates, x)
        return new_pointset


class seg_network(nn.Module):
    def __init__(self, K_seg):
        super(seg_network, self).__init__()
        self.set_abstraction_1 = SetAbstraction(512, 0.2, [64, 64, 128], 0)
        self.set_abstraction_2 = SetAbstraction(128, 0.4, [128, 128, 256], 128)
        self.global_set_abstraction = GlobalSetAbstraction([256, 512, 1024], 256)
        self.feature_propagation1 = FeaturePropagation([256, 256], 1024+256, False)
        self.feature_propagation2 = FeaturePropagation([256, 128], 256+128, False)
        self.feature_propagation3 = FeaturePropagation([128, 128, 128, 128, K_seg], 128, True)

    def forward(self, pointset):
        pointset1 = self.set_abstraction_1.abstract(pointset)
        pointset2 = self.set_abstraction_2.abstract(pointset1)
        pointset3 = self.global_set_abstraction.abstract(pointset2)
        pointset4 = self.feature_propagation1(pointset3, pointset2)
        pointset5 = self.feature_propagation2(pointset4, pointset1)
        pointset6 = self.feature_propagation3(pointset5, pointset)
        return pointset6


def load_data():
    classes = [0]
    data_train = dataset.Dataset(root=rootDataset, dataset_name='shapenetpart', num_points=2048, split='train', segmentation=True)
    data_test = dataset.Dataset(root=rootDataset, dataset_name='shapenetpart', num_points=2048, split='test', segmentation=True)
    data_val = dataset.Dataset(root=rootDataset, dataset_name='shapenetpart', num_points=2048, split='val', segmentation=True)

    data_list = [data_train, data_test, data_val]
    processed_data = []
    for data in data_list:
        processed_data.append([])
        for object in data:
            category = object[1].detach().numpy()[0]
            if category in classes:
                if len(processed_data[0])==0:
                    print("Name of the class: ", object[-2])
                coordinates = object[0]
                features = torch.zeros(coordinates.shape[0], 0)
                label = object[2]
                pointset = PointSet(coordinates, features)
                obj = [pointset, label]
                processed_data[-1].append(obj)
    print("Data loaded")
    return processed_data


def train():
    data_list = load_data()
    train_data = data_list[0]
    test_data = data_list[1]
    val_data = data_list[2]
    K_seg = 50
    model = seg_network(K_seg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loss_list = []
    for epoch in range(1):
        print("Epoch: ", epoch)
        model.train()
        L = len(train_data)
        for i, data in enumerate(train_data):
            optimizer.zero_grad()
            pointset = data[0]
            label = data[1]
            result = model(pointset)
            pred = result.features
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            print("Loss: ", loss.item(), " - Progress: ", i/L*100, "%")
            loss_list.append(loss.item())
        test(model)
    #plot loss
    plt.plot(loss_list)
    plt.savefig("loss.png")
    plt.show()
    #save model
    torch.save(model.state_dict(), "model.pt")
    print("Model saved under model.pt")
    ##test on one object
    model.eval()
    test_object = val_data[10]
    pointset = test_object[0]
    label = test_object[1]
    result = model(pointset)
    pred = result.features
    pred = pred.detach().numpy()
    pred = np.argmax(pred, axis = 1)
    print(pred)
    print(label)
    #plot
    fig = plt.figure()
    ax, ax2 = fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')
    ax.scatter(pointset.coordinates[:,0], pointset.coordinates[:,1], pointset.coordinates[:,2], c = pred)
    ax.set_title("Prediction")
    ax2.scatter(pointset.coordinates[:,0], pointset.coordinates[:,1], pointset.coordinates[:,2], c = label)
    ax2.set_title("Ground truth")
    plt.show()
    
def eval():
    data_list = load_data()
    train_data = data_list[0]
    test_data = data_list[1]
    val_data = data_list[2]
    K_seg = 50
    model = seg_network(K_seg)
    model.load_state_dict(torch.load("model_custom_airplane.pt"))
    model.eval()
    #evaluate on validation set
    correct = 0
    total = 0
    with torch.no_grad():
        l = len(val_data)
        for i, data in enumerate(val_data):
            pointset = data[0]
            label = data[1]
            label = label.detach().numpy()
            result = model(pointset)
            pred = result.features
            pred = pred.detach().numpy()
            pred = np.argmax(pred, axis = 1)
            correct += np.sum(pred == label)
            total += label.shape[0]
            print("Progress: ", i/l*100, "%")
            print("Current accuracy: ", correct/total)
    print("Accuracy: ", correct/total)
    print("correct: ", correct, " - total: ", total)

def test(model=None):
    data_list = load_data()
    train_data = data_list[0]
    test_data = data_list[1]
    val_data = data_list[2]
    K_seg = 50
    if model is None:
        model = seg_network(K_seg)
        model.load_state_dict(torch.load("model_custom_airplane.pt"))
    model.eval()
    test_object = val_data[2]
    pointset = test_object[0]
    coords = pointset.coordinates.detach().numpy()
    label = test_object[1]
    label = label.detach().numpy()
    result = model(pointset)
    pred = result.features
    pred = pred.detach().numpy()
    #argmax
    pred = np.argmax(pred, axis = 1)
    #precision
    correct = np.sum(pred == label)
    total = label.shape[0]
    print("Accuracy: ", correct/total)
    #plot
    fig = plt.figure()
    ax, ax2 = fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax2.set_xlim(-1,1)
    ax2.set_ylim(-1,1)
    ax2.set_zlim(-1,1)
    ax.view_init(elev=-44, azim=60)
    ax2.view_init(elev=-44, azim=60)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], c = pred)
    ax.set_title("Prediction")
    ax2.scatter(coords[:,0], coords[:,1], coords[:,2], c = label)
    ax2.set_title("Ground truth")
    fig.set_tight_layout(True)
    plt.savefig("test.png")
    plt.show()


if __name__ == '__main__':
    #train()
    #eval()
    test()
