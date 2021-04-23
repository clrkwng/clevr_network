import sys
sys.path.insert(0, 'model/')
from model_v3 import *
sys.path.pop(0)
sys.path.insert(0, 'data_processing/')
from clevr_dataset import *
sys.path.pop(0)

import torch.optim as optim
import math

BATCH_SIZE = 128
NUM_BATCHES = math.ceil(1.0 * 10000 / BATCH_SIZE)

def train(model, criterion, optimizer, train_loader, valid_loader, save_model_path, num_epochs):
	epochs_no_improve = 0
	valid_loss_min = np.Inf
	valid_max_acc = 0
	history = []

	t = tqdm(range(1, num_epochs + 1), miniters=100)
	step = 0

	for epoch in t:
		total_loss = 0
		cube_correct = 0
		cylinder_correct = 0
		sphere_correct = 0

		for i, data in enumerate(train_loader, 0):
			model.train()
			inputs, labels = data
			inputs = inputs.cuda()
			cube_labels, cylinder_labels, sphere_labels = labels[:,0].cuda(), labels[:,1].cuda(), labels[:,2].cuda()

			# Forward + Backward + Optimize
			optimizer.zero_grad()
			cube_preds, cylinder_preds, sphere_preds = model(inputs)
			cube_loss = criterion(cube_preds, cube_labels)
			cylinder_loss = criterion(cylinder_preds, cylinder_labels)
			sphere_loss = criterion(sphere_preds, sphere_labels)
			loss = cube_loss + cylinder_loss + sphere_loss
			loss.backward()
			optimizer.step()

			cube_correct += get_num_correct(cube_preds, cube_labels)
			cylinder_correct += get_num_correct(cylinder_preds, cylinder_labels)
			sphere_correct += get_num_correct(sphere_preds, sphere_labels)
			total_loss += loss.item()

			step += 1
		
		cube_acc = round(cube_correct/10000, 6)
		cylinder_acc = round(cylinder_correct/10000, 6)
		sphere_acc = round(sphere_correct/10000, 6)
		epoch_loss = round(total_loss/NUM_BATCHES, 6)
		t.set_description(f"Epoch: {epoch}/{num_epochs}, Loss: {epoch_loss}, Cube acc: {cube_acc},\
												Cylinder acc: {cylinder_acc}, Sphere acc: {sphere_acc}")

def main():
	model = WangNetV3(ResBlock, [1], 3).cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-2)

	train_dataset = CLEVRDataset('../clevr-dataset-gen/output/train/')
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

	valid_loader = 2
	save_model_path = 2
	num_epochs = 100

	train(model, criterion, optimizer, train_loader, valid_loader, save_model_path, num_epochs)

if __name__ == "__main__":
	main()