import random
import torch
import sys
class Loader():
	def __init__(self,data,start=0,end=sys.maxsize,batch_size=5,shuffle=False):
		'''
		@param data a torch.utils.data.Dataset instance
		@param start the start point of the dataset you want to load
		@end the end point of the dataset you want to load
		@batch_size the size of each batch given by next(iterator)
		@shuffle a boolean to indicate whether to shuffle the data or not
		'''
		self.data=data
		self.num_iter=int((end-batch_size-start)/batch_size)
		self.batch_size=batch_size
		self.shuffle=shuffle

		self.start=self.getInRange(start)
		self.end=self.getInRange(end)

	def __iter__(self):
		return loader(self.data,self.start,self.end,self.batch_size,self.shuffle)
	
	def __len__(self):
		return self.end-self.start

	def getInRange(self,value):
		#return the possible range given by start point and end point
		value=min(len(self.data),value)
		value=max(value,0)

		return value

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

		img=[self.data[j][0] for j in range(self.List[self.i][0],self.List[self.i][1])]

		img=torch.stack(img)

		label=[self.data[j][1] for j in range(self.List[self.i][0],self.List[self.i][1])]

		self.i+=1
		return img,label
	
	def returnList(self,start,end,batch_size,shuffle):
		List=list(range(start,end-batch_size,batch_size))
		#List:[(start_point,start_point+batch_size)]
		List=[(List[i],List[i]+batch_size) for i in range(len(List))]

		#Get the rest of the data
		if (end-batch_size-start)%batch_size!=0:
			List.append((List[-1][1],end))

		if shuffle:
			random.shuffle(List)

		return List

