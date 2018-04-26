from common import *
from utils import *
from loader import *
from models import *
from loss import *

def run_train():

	traindata = ADNIDataloader(data_file=FILE_TRAIN, data_path=data_path, transform=ToTensor())
	#valdata = ADNIDataloader(data_file=FILE_VAL, data_path=data_path, transform=ToTensor())
	testdata = ADNIDataloader(data_file=FILE_TEST, data_path=data_path, transform=ToTensor())


	train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(testdata, batch_size=batch_size) # num_workers=pars['num_workers']


	model = Autoencoder_bottle(100)
	# model = Autoencoder_bottle_wide(bottlerange=2)
	# model = Autoencoder()
	model.cuda()
	model.train(True)

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	criterion = nn.MSELoss()  # TO CHANGE!
	# criterion = nn.L1Loss()
	# criterion = nn.CrossEntropyLoss()
	
	for epoch in range(num_epochs):
		for idx,data in enumerate(train_loader):

			imgs = Variable(data).cuda()
			output = model(imgs)
			loss = criterion(output, imgs)
			loss.backward()
			optimizer.step()

			if idx%500 == 0:
				print ('Check train loss {:5} : {:.5f}'.format(idx, loss.data.cpu().numpy()[0]))

		print ('end of epoch!\n')

		if epoch % export_checkpoint == 0:
			pic = to_img(imgs.cpu().data)
			save_image(pic, '{}input_epoch{}.png'.format(image_dump_path,epoch))
			pic = to_img(output.cpu().data)
			save_image(pic, '{}output_epoch{}.png'.format(image_dump_path,epoch))




if __name__ == '__main__':

	print ('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 30))


	run_train()

