import random
import torch
class Loader():
	def __init__(self,data,start=0,end=-1,batch_size=5,shuffle=False):
		self.data=data
		self.num_iter=int((end-batch_size-start)/batch_size)
		self.batch_size=batch_size
		self.shuffle=shuffle
		self.start=start
		if end==-1:
			self.end=len(data)
		else:
			self.end=end
	def __iter__(self):
		return loader(self.data,self.start,self.end,self.batch_size,self.shuffle)
	
	def __len__(self):
		return self.end-self.start

class loader():
	def __init__(self,data,start,end,batch_size,shuffle=False):
		self.List=self.returnList(start,end,batch_size,shuffle)
		self.num_iter=len(self.List)
		self.batch_size=batch_size
		self.data=data
		self.i=0
	def __iter__(self):
		return self
	def __next__(self):
		if(self.i>len(self.List)-1):
			raise StopIteration()

		img=[self.data[j][0] for j in range(self.List[self.i],self.List[self.i]+self.batch_size)]    
		img=torch.stack(img)
		label=[self.data[j][1] for j in range(self.List[self.i],self.List[self.i]+self.batch_size)]

		self.i+=1
		return img,label
	
	def returnList(self,start,end,batch_size,shuffle):
		List=list(range(start,end-batch_size,batch_size))
		if shuffle:
			random.shuffle(List)
		return List;

