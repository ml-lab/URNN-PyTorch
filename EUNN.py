import torch
from torch.autograd import Variable
import torch.nn as nn


class EUNN(nn.Module):

	def __init__(self, N, L):
		super(EUNN, self).__init__()


		self.L = L
		self.N = N

		self.thetaA = nn.Parameter(torch.rand(L/2, N/2)) 
		self.thetaB = nn.Parameter(torch.rand(L/2, N/2 - 1)) 

		sinA = torch.sin(self.thetaA)
		cosA = torch.cos(self.thetaA)
		sinB = torch.sin(self.thetaB)
		cosB = torch.cos(self.thetaB)

		I = Variable(torch.ones((L/2, 1)))
		O = Variable(torch.zeros((L/2, 1)))

		self.diagA = torch.stack((cosA, cosA), 2)
		self.offA = torch.stack((-sinA, sinA), 2)
		self.diagB = torch.stack((cosB, cosB), 2)
		self.offB = torch.stack((-sinB, sinB), 2)

		self.diagA = self.diagA.view(L/2, N)
		self.offA = self.offA.view(L/2, N)
		self.diagB = self.diagB.view(L/2, N-2)
		self.offB = self.offB.view(L/2, N-2)

		self.diagB = torch.cat((I, self.diagB, I), 1)
		self.offB = torch.cat((O, self.offB, O), 1)

		# print self.diagA, self.offA, self.diagB, self.offB

	def forward(self, x):

		batch_size = x.size()[0]
		for i in range(self.L/2):
			# A
			y = x.view(batch_size, self.N/2, 2)
			y = torch.stack((y[:,:,1], y[:,:,0]), 2)
			y = y.view(batch_size, self.N)

			x = torch.mul(x, self.diagA[i].expand_as(x))
			y = torch.mul(y, self.offA[i].expand_as(x))

			x = x + y

			# B
			x_top = x[:,0]
			x_mid = x[:,1:-1].contiguous()
			x_bot = x[:,-1]
			y = x_mid.view(batch_size, self.N/2-1, 2)
			y = torch.stack((y[:, :, 1], y[:, :, 0]), 1)
			y = y.view(batch_size, self.N-2)
			x_top = torch.unsqueeze(x_top, 1)
			x_bot = torch.unsqueeze(x_bot, 1)
			# print x_top.size(), y.size(), x_bot.size()
			y = torch.cat((x_top, y, x_bot), 1)

			x = x * self.diagB[i].expand(batch_size, self.N)
			y = y * self.offB[i].expand(batch_size, self.N)

			x = x + y		
		

		return x

if __name__ == '__main__':
	net = EUNN(400, 200)

	params = list(net.parameters())
	print(len(params))

	batch_size = 400

	inp = Variable(torch.randn(batch_size, 400))
	out = net(inp)

	a = torch.sum(inp * inp)
	b = torch.sum(out * out)
	print a, b, (a-b)/a
