import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init


class GORUCell(nn.Module):
	"""An GORU cell."""

	def __init__(self, input_size, hidden_size, capacity):

		super(GORUCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.capacity = capacity
		self.U = nn.Parameter(
			torch.FloatTensor(input_size, hidden_size))
		self.thetaA = nn.Parameter(
			torch.FloatTensor(hidden_size//2, capacity//2))
		self.thetaB = nn.Parameter(
			torch.FloatTensor(hidden_size//2-1, capacity//2))
		self.bias = nn.Parameter(
			torch.FloatTensor(hidden_size))

		self.gate_U = nn.Parameter(
			torch.FloatTensor(input_size, 2 * hidden_size))
		self.gate_W = nn.Parameter(
			torch.FloatTensor(hidden_size, 2 * hidden_size))
		self.gate_bias = nn.Parameter(torch.FloatTensor(2 * hidden_size))

		self.reset_parameters()

	def reset_parameters(self):
		"""
		Initialize parameters  TO DO
		"""
		init.uniform(self.thetaA, a=-0.1, b=0.1)
		init.uniform(self.thetaB, a=-0.1, b=0.1)
		init.uniform(self.U, a=-0.1, b=0.1)
		init.orthogonal(self.gate_U.data)
		
		gate_W_data = torch.eye(self.hidden_size)
		gate_W_data = gate_W_data.repeat(1, 2)
		self.gate_W.data.set_(gate_W_data)

		init.constant(self.bias.data, val=0)
		init.constant(self.gate_bias.data, val=0)

	def _EUNN(self, hx, thetaA, thetaB):

		L = self.capacity
		N = self.hidden_size

		sinA = torch.sin(self.thetaA)
		cosA = torch.cos(self.thetaA)
		sinB = torch.sin(self.thetaB)
		cosB = torch.cos(self.thetaB)

		I = Variable(torch.ones((L//2, 1)))
		O = Variable(torch.zeros((L//2, 1)))

		diagA = torch.stack((cosA, cosA), 2)
		offA = torch.stack((-sinA, sinA), 2)
		diagB = torch.stack((cosB, cosB), 2)
		offB = torch.stack((-sinB, sinB), 2)

		diagA = diagA.view(L//2, N)
		offA = offA.view(L//2, N)
		diagB = diagB.view(L//2, N-2)
		offB = offB.view(L//2, N-2)

		diagB = torch.cat((I, diagB, I), 1)
		offB = torch.cat((O, offB, O), 1)

		batch_size = hx.size()[0]
		x = hx
		for i in range(L//2):
# 			# A
			y = x.view(batch_size, N//2, 2)
			y = torch.stack((y[:,:,1], y[:,:,0]), 2)
			y = y.view(batch_size, N)

			x = torch.mul(x, diagA[i].expand_as(x))
			y = torch.mul(y, offA[i].expand_as(x))

			x = x + y

			# B
			x_top = x[:,0]
			x_mid = x[:,1:-1].contiguous()
			x_bot = x[:,-1]
			y = x_mid.view(batch_size, N//2-1, 2)
			y = torch.stack((y[:, :, 1], y[:, :, 0]), 1)
			y = y.view(batch_size, N-2)
			x_top = torch.unsqueeze(x_top, 1)
			x_bot = torch.unsqueeze(x_bot, 1)
			# print x_top.size(), y.size(), x_bot.size()
			y = torch.cat((x_top, y, x_bot), 1)

			x = x * diagB[i].expand(batch_size, N)
			y = y * offB[i].expand(batch_size, N)

			x = x + y
		return x	

	def _modReLU(self, h, bias):
		"""
		sign(z)*relu(z)
		"""
		batch_size = h.size(0)
		sign = torch.sign(h)
		bias_batch = (bias.unsqueeze(0)
					  .expand(batch_size, *bias.size()))
		return sign * functional.relu(torch.abs(h) + bias_batch)

	def forward(self, input_, hx):
		"""
		Args:
			input_: A (batch, input_size) tensor containing input
				features.
			hx: initial hidden, where the size of the state is
				(batch, hidden_size).

		Returns:
			newh: Tensors containing the next hidden state.
		"""
		batch_size = hx.size(0)
		
		bias_batch = (self.gate_bias.unsqueeze(0)
					  .expand(batch_size, *self.gate_bias.size()))
		gate_Wh = torch.addmm(bias_batch, hx, self.gate_W)
		gate_Ux = torch.mm(input_, self.gate_U)		
		r, z = torch.split(gate_Ux + gate_Wh,
								 split_size=self.hidden_size, dim=1)

		Ux = torch.mm(input_, self.U)
		unitary = self._EUNN(hx=hx, thetaA=self.thetaA, thetaB=self.thetaB)
		unitary = unitary * r
		newh = Ux + unitary
		newh = self._modReLU(newh, self.bias)		
		newh = hx * z + (1-z) * newh

		return newh

	def __repr__(self):
		s = '{name}({input_size}, {hidden_size})'
		return s.format(name=self.__class__.__name__, **self.__dict__)

		
class GORU(nn.Module):

	"""A module that runs multiple steps of GORU."""

	def __init__(self, input_size, hidden_size, num_layer, cell_class=GORUCell, capacity=2,
				 use_bias=True, batch_first=False, dropout = 0, **kwargs):
		super(GORU, self).__init__()
		self.cell_class = cell_class
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layer = num_layer
		self.capacity = capacity
		self.batch_first = batch_first
		self.dropout = dropout

		self.cells = []
		for layer in range(num_layer):
			layer_input_size = input_size if layer == 0 else hidden_size
			cell = cell_class(input_size=layer_input_size,
							  hidden_size=hidden_size,
							  capacity=capacity,
							  **kwargs)
			self.cells.append(cell)
			setattr(self, 'cell_{}'.format(layer), cell)
		self.dropout_layer = nn.Dropout(dropout)
		self.reset_parameters()

	def reset_parameters(self):
		for cell in self.cells:
			cell.reset_parameters()

	@staticmethod
	def _forward_rnn(cell, input_, length, hx):
		max_time = input_.size(0)
		output = []
		for time in range(max_time):
			h_next = cell(input_=input_[time], hx=hx)
			# mask = (time < length).float().unsqueeze(1).expand_as(h_next)
			# h_next = h_next*mask + hx*(1 - mask)
			output.append(h_next)
		output = torch.stack(output, 0)
		return output, h_next

	def forward(self, input_, length=None, hx=None):
		if self.batch_first:
			input_ = input_.transpose(0, 1)
		max_time, batch_size, _ = input_.size()
		if length is None:
			length = Variable(torch.LongTensor([max_time] * batch_size))
			if input_.is_cuda:
				length = length.cuda()
		if hx is None:
			hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
		
		h_n = []
		layer_output = None
		for layer in range(self.num_layer):
			layer_output, layer_h_n = GORU._forward_rnn(
				cell=self.cells[layer], input_ = input_, length=length, hx =hx)
			input_ = self.dropout_layer(layer_output)
			h_n.append(layer_h_n)
		output=layer_output
		h_n = torch.stack(h_n, 0)	
		return output, h_n

