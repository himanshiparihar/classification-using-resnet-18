import numpy as np 
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F 
from torchvision import datasets , transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import sys
print(sys.path)
sys.path.append('/lib/python3.7/site-packages')
from PIL import Image




def setparm(model , extracting):
	if extracting:
		for par in model.parameters():
			par.require_grad = False


def trainmod(model ,train_loader , test_loader , criterion ,optimizer, epochs = 25):
	steps = 0
	running_loss = 0
	print_every = 10
	train_losses, test_losses = [], []

	for epoch in range(epochs):
		for ip , lb in train_loader:
			steps=+ 1
			ip , lb = ip.to(device),lb.to(device)
			optimizer.zero_grad()
			logps = model.forward(ip)
			loss = criterion(logps , lb)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

			if steps % print_every == 0:
				test_loss = 0
				accuracy = 0
				model.eval()
				with torch.no_grad():
					for inputs, labels in testloader:
						inputs, labels = inputs.to(device),labels.to(device)
						logps = model.forward(inputs)
						batch_loss = criterion(logps, labels)
						test_loss += batch_loss.item()
					
						ps = torch.exp(logps)
						top_p, top_class = ps.topk(1, dim=1)
						equals = top_class == labels.view(*top_class.shape)
						accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
				train_losses.append(running_loss/len(trainloader))
				test_losses.append(test_loss/len(testloader))                    
			# print(f"Epoch {epoch+1}/{epochs}.. "
			# 	  f"Train loss: {running_loss/print_every:.3f}.. "
			# 	  f"Test loss: {test_loss/len(testloader):.3f}.. "
			# 	  f"Test accuracy: {accuracy/len(testloader):.3f}")
				print("epoch , train_losses , test_loss , test_accuracy " , epoch+1/epochs , running_loss/print_every ,test_loss/len(testloader) ,accuracy/len(testloader))
				running_loss = 0
				model.train()
			
	torch.save(model, 'aerialmodel.pth')

if __name__ == '__main__':
	datadir = 'hymenoptera_data'

	def splitTrainTest (datadir , valid_size = .2):
		
		train_trans= transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
		test_trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

		train_data = datasets.ImageFolder(datadir , transform = train_trans)
		test_data = datasets.ImageFolder(datadir , transform = test_trans)

		lent = len(train_data) # for spliting
		# print('length',lent)
		indices = list(range(lent)) # for shuffling
		# print("indices",indices)
		
		split = int(np.floor(valid_size * lent))
		# print("split",split)
		np.random.shuffle(indices)

		train_idx , test_idx = indices[split:] , indices[:split]

		train_sampler = SubsetRandomSampler(train_idx)
		test_sampler = SubsetRandomSampler(test_idx)

		train_loader = torch.utils.data.DataLoader(dataset=train_data, sampler=train_sampler, batch_size=64)
		test_loader = torch.utils.data.DataLoader(dataset=train_data, sampler=test_sampler, batch_size=64) 

		return train_loader , test_loader

	train_loader , test_loader = splitTrainTest (datadir , .2)
	print(train_loader.dataset.classes)
	device = torch.device("cpu")

	model = models.resnet18(pretrained = True)

	setparm(model,True)
	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features , 2)
	

	# freezing layers to stop back propogation
	# for para in model.parameters():
	# 	para.require_grad = False


	# model.fc = nn.Sequential(nn.Linear(2048 , 512) , nn.ReLU(),nn.Dropout(0.2),nn.Linear(512 ,2),nn.LogSoftmax(dim=1))
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
	model.to(device)

	params_to_update = []
	for name , param in model.named_parameters():
		if param.requires_grad is True:
			params_to_update.append(param)
			print('/t',name)
	trainmod(model , train_loader , test_loader , criterion , optimizer)

