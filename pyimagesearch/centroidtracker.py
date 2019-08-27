#import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

peoplecount=0

class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		#print("object id",self.nextObjectID)
		#print("objects",self.objects)
		#print("disappear",self.disappeared)

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared
		#print("maxdisappear",self.maxDisappeared)

		# store the maximum distance between centroids to associate
		# an object -- if the distance is larger than this maximum
		# distance we'll start to mark the object as "disappeared"
		self.maxDistance = maxDistance
		#print("maxdistance",self.maxDistance)

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid  # here, the centroid with ID as key stored.
		self.disappeared[self.nextObjectID] = 0 # if object detected, then it's disappear value sets as 0 
		self.nextObjectID += 1 # the object id incremented from current id
		#print("centroid",self.objects)
		#print("regdisappear",self.disappeared)
		#print("Total number of peoples",len(self.objects))
		

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		#print("Deleted",objectID)
		#print("id",self.objects)
		del self.objects[objectID] # object deleted with the current ID
		del self.disappeared[objectID] # also deleted from disappeared.
		#print(self.objects)
		

	def update(self, rects): # rect means , the bounding box 
		# check to see if the list of input bounding box rectangles
		# is empty
		#print("rects",rects) 
		#print("length of rect",len(rects))
		#print("Total number of peoples = ",len(self.objects))
		globals()['peoplecount'] = len(self.objects)
		if len(rects) == 0:  # if there is no bounding box the loop will works
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()): # here,loop started with existing obj ID's.
				#print("objID Update",objectID)
				#print("disappeared update",self.disappeared)
				self.disappeared[objectID] += 1 # disappeared incremented for current obj ID
				

				if self.disappeared[objectID] > self.maxDisappeared: # when the disappear greater than maxdisappear
					self.deregister(objectID) # deregister function calls and deleted the obj ID from both object and disapppear dictionary.

			# return early as there are no centroids or tracking info
			# to update
			return self.objects # the updated results returned
		

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int") # inintially , it is [0,0]. it is for storing centroids of objects
		#print("input centroids",inputCentroids)
		#print("rl = ",len(rects))


		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects): # here, i consist of the object id, startX,startY,endX,endY consist of co-ordinate value of bounding box in rect
			#print("i=",i)
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			#print("cX= {} cY={}".format(cX,cY))
			inputCentroids[i] = (cX, cY) # centroid value stored(we gets, (x,y)  )
			#print("cetroid of I",inputCentroids[i])
		#print("length of objects",len(self.objects))

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0: # if the object is not registered
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
				#print("added =",i)
				#print("objcen = ",len(inputCentroids))
				#print("objcentro = ",inputCentroids[i])
				

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys()) # existing object ID's fetched.
			objectCentroids = list(self.objects.values()) # existing obj  ID's centroid fetched
			#print("secobjID = ",objectIDs)
			#print("secobjVal =",objectCentroids)

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			#print("Distance = ",D)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()
			#print("rows min =",rows)

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()
			#print("Total number of peoples 4",len(self.objects))

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in usedRows or col in usedCols:
					continue

				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object
				if D[row, col] > self.maxDistance:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects
