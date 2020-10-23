#Test cases
boxes=np.array([[0,30,10,50],[0,40,1,41],[5,45,6,46],[15,40,16,41],[20,10,30,20],[20,40,30,60]])
labels=np.array([1,2,2,2,1,1,1])
scores=np.array([0.9,0.3,0.6,0.9,0.3,0.5])
box,label,score=filter(boxes,labels,scores)
print("boxes:",box,"\nlabels:",label,"\nscore:",score)