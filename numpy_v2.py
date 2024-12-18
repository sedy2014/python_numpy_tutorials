
from __future__ import print_function
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return - (2/9) * x ** (-5/3)

#*****import numpy as np
import matplotlib.pyplot as plt
x = np.array([11 ,12 ,6 ,15, 6, 12, 14, 11, 4, 4 ,7 ,7 ,4 ,5 ,8 ,13, 13, 9, 7 ,8 ,6 ,12 ,16 ,11, 8, 4, 7, 8, 11, 9, \
11, 12, 13, 14, 12, 9 ,8 ,6 ,4 ,6 ,15, 7 ,8 ,5 ,5 ,10, 12, 5, 19, 12, 14, 10, 13 ,8 ,9 ,9 ,14, 12, 13, 16, 8 ,5 ,6 ,11, 7, 7, \
11, 4, 14, 15, 9 ,10 ,12 ,10 ,10 ,7 ,15 ,8 ,6 ,3 ,13, 5, 5 ,13, 15, 12, 11, 8 ,6 ,7 ,10, 10, 11, 7 ,9 ,12, 13, 4 ,4,5])

print(len(x))
bins = np.arange(20)-0.5
hist,edges = np.histogram(x,bins=bins)


xUniq = np.unique(x)
xCnt = np.zeros(len(xUniq))
for i in range(len(xUniq)):
    xCnt[i] = np.sum(np.isin(x, xUniq[i]))

# plt.subplot(2,1,1)
# plt.xlabel('Rooms')
# plt.ylabel("Frequency of People responded")
# plt.title("Histogram of rooms")
# plt.subplot(2,1,2)
plt.scatter(xUniq,xCnt,cmap="Greys")
plt.xlabel('Number of Rooms')
plt.ylabel("Number  of People responded")
plt.xticks(xUniq)
plt.show()

y = np.bincount(x)
y= np.arange(1,hist.max()+1)
x = np.arange(19)
X,Y = np.meshgrid(x,y)
plt.scatter(X,Y,c=Y<=hist,cmap="Greys")
plt.show()
#******Containers********************
# Containers are data structures holding elements
# and typically hold all their values in memory,e.g lists,dict,tuple
# Technically, an object is a container when it can be asked whether it contains a certain element
# most containers are also iterable

# *************** Iterables *********************
# An iterable is any object, not necessarily a data structure,that can be iterated over,
# meaning you can loop over its elements.
#Examples of iterables include lists, tuples, strings, dictionaries, sets,
# iterator is an object which keeps a state and produces the next value each time it is iterated upon.
# Iterable  generates an Iterator when using  iter() method == It implement the __iter__() method,
# thus returning an iterator
#If an object is iterable, it can be passed to the built-in Python function iter()
# Iterator  iterate over an iterable object using  __iter__ and __next__() method.
# Examples of iterators include generators (created using functions with yield),
# enumerate(), zip(), and map()
# All iterators are iterables, but not all iterables are iterators.

## list of cities
cities = [1,2,3]
# initialize iterator object
iterator_obj = iter(cities) # type(iterator_obj): list_iterator
elem1 = next(iterator_obj)
elem2 = next(iterator_obj)
elem3 = next(iterator_obj)

# If you want to grab all the values from an iterator at once, you can use the built-in list()/tuple function. Among other possible uses,
# list() takes an iterator as its argument, and returns a list consisting of all the values that the iterator yielded:
# for example, build infinite iterator for odd numbers
iterator_obj1 = iter(cities)
all_elem = list(iterator_obj1)
# Part of the elegance of iterators is that they are “lazy.” That means that when you create an iterator,
# it doesn’t generate all the items it can yield just then.
# It waits until you ask for them with next(). Items are not created until they are requested.

# create class that has _iter_ and _next_ methods, and goes over odd elemnts
class OddIter:
    def __iter__(self):
        # initial elemnt to start with
        self.num = 1
        return self

    def __next__(self):
        num = self.num
        self.num += 2
        return num

#  create iterator now
a1 = iter(OddIter())
e1 = next(a1)
e2 = next(a1)
# Be careful to include a terminating condition, when iterating over these type of infinite iterators.

#*** class that uses iterator to  extract elements at odd index from list
class OddIter1:
    def __init__(self, data):
        self.data = data
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        while self.index < len(self.data):
            self.index += 2
            return self.index
        raise StopIteration()

data = [1, 3,5,8,10,7,2,11,13]
# odd_iterator1 = OddIter1(data)
# for odd_indx in odd_iterator1:
#     print(data[odd_indx])

#*** class that uses iterator to  extract odd elements from list
class OddIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.index < len(self.data):
            if self.data[self.index] % 2 != 0:
                result = self.data[self.index]
                self.index += 1
                return result
            else:
                self.index += 1
        raise StopIteration()


odd_iterator = OddIterator(data)
for odd_number in odd_iterator:
    print(odd_number)

#************* Generators*******************
#A generator allows you to create iterators , in an elegant succinct syntax
# that avoids writing classes with __iter__() and __next__() methods.


# generator function
# uses the Python yield keyword instead of return
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1
# yield indicates where a value is sent back to the caller, but unlike return, you don’t exit the function afterward.
# Instead, the state of the function is remembered. That way, when next() is called on a generator object
# (either explicitly or implicitly within a for loop),
# the previously yielded variable num is incremented, and then yielded again

# usage:
gen = infinite_sequence()
e1 = next(gen)
e2 = next(gen)
e3 = next(gen)

#********* generator expression:allow you to quickly create a generator object in just a few lines of code. **********
# advantage: you can create them without building and holding the entire object in memory before iteration
# ****# for example:
# write a expression to square set of numbers
# using list
nums_squared_lc = [num**2 for num in range(5)]
# using generator by using ()
nums_squared_gc = (num**2 for num in range(5))
nums_squared_lc1 = list(nums_squared_lc)

# ******************different forms of for loop************************

# When a for loop is executed, for statement calls iter() on the object, which it is supposed to loop over.
# If this call is successful, the iter call will return an iterator object that defines the method __next__(),
# which accesses elements of the object one at a time

#for <var> in <iterable>:
   # <statement(s)>



# python range :The range type represents an immutable sequence of numbers( is a type like list and tuple)
print("***************FOR LOOP **********************")
rng1 = list(range(5)) ## Prints out 3,4,5
rng2 = list(range(3, 8, 2)) # range(st,stop,delta): 3,5,7

for i in np.arange(4):
    print(i)

# uses array as iterable seq
for i in np.array([8,9,10]):
    print(i)

 #  uses range as iterable seq
for i in range(3, 8, 2):
    print(i)
#** redo above with list comprehension
# newlist = [expression for item in iterable if condition == True]
[print(i) for i in np.arange(4)]
[print(i) for i in np.array([8,9,10])]
[print(i) for i in range(3, 8, 2)]
#extract only oddd elemnts
[print("only odd elemnts" + str(i)) for i in np.arange(10) if i%2!=0]

# using enumerate:Accessing each item in a list (or another iterable),Also getting the index of each item accessed
start_indx =2 # will aceess each elemnt, starting counter at this index number
itm1 = [8,9,10]
for indx,i in enumerate(itm1,start_indx): #
    print(indx,i) # prints 0,8..1,9...2,10, but with start_indx=2, gives 2,8..3,9..4,10

# numpy  np.ndenumerate(arr):  iterator yielding pairs of array coordinates and values for ND array

nd1 = np.array([[1, 2], [3, 4]])
for index, x in np.ndenumerate(nd1):
    print(index, x)
#(0, 0) 1
#(0, 1) 2
#(1, 0) 3
#(1, 1) 4
# # can be used with 1 d array like enumerate too.
[print (index,i) for index,i in np.ndenumerate([8,9,10])]
#************* python zip: similar to enumerate but is most useful for acessing multiple lists *************
# returns an iterator of tuples based on the iterable objects.
# If a single iterable is passed, zip() returns an iterator of tuples with each tuple having only one element.
# If multiple iterables are passed, zip() returns an iterator of tuples with each tuple having elements from all the iterables.
group = ['A','B','C']
tag = ['a','b','c']
for idx, x in enumerate(group):
    print(x, tag[idx])
# prints
# A a
# B b
# C c
# same can be accompolished by zip

for x, y in zip(group, tag):
    print(x, y)
# Note: Izip is similar to zip, but zip combines all elements at one time(more memory), but izip does that one by one

#

#****************** creating array inside loop ****
# python append and extend
# append: Appends object at the end of LIST.( length of list only increases by 1)
# extend:Extends list by appending elements from the iterable (( length of list increseas by num of elements in iterable)
x = [1, 2, 3]
x.append([4, 5])  # [1,2,3,[4,5]]
x1 = [1, 2, 3]
x1.extend([4, 5])
x2 = [1, 2, 3]
x2 =  x2 + [4,5]  # same as extend

#****** however, numpy append will append to an array/list: np.append(arr or even list,elements to append,axis=..)

#
# create array [0,1,2,3,4] inside loop
x_arr_ins_loop = np.array([])
for i in range(4):
    x_arr_ins_loop = np.append(x_arr_ins_loop, i)

# *********dictionary   ***********
# empty dict
dic_empty =  dict()
# created with {key1:val1,key2:val2..}
dic1 = {1:'a',2:'b',3:'c'}
# key with multiple values
dic2 = {1:['a','b'],2:['c','d']}
# dict using list
y = {1:[x for x in range(5)],2:[x for x in range(10)]}

# items() gives class dict_items , which you can't acess individually, as is but
# can access within for loop
items1 = dic1.items()
print(items1)
print ("Dict key-value are : ")
for k1, v1 in dic1.items():
    print (k1, v1)
# gives 1 a
      # 2 b
      # 3 c
#** ***create list of dict keys and values from dictionary
def read_dict(dic):
    k2_arr = np.array([])
    v2_arr = np.array([])
    v2_list = []
    for k2, v2 in dic.items():
        k2_arr = np.append(k2_arr,k2)
        v2_arr = np.append(v2_arr, v2)
        v2_list.append(v2)
    return k2_arr,v2_arr,v2_list
k2_arr,v2_arr,v2_list = read_dict(dic2)

# recerate above using class
class DictReader:
    def __init__(self):
        pass

    def read_dict(self, dic):
        self.k2_arr = np.array([])
        self.v2_arr = np.array([])
        self.v2_list = []
        for k2, v2 in dic.items():
            self.k2_arr = np.append(self.k2_arr, k2)
            self.v2_arr = np.append(self.v2_arr, v2)
            self.v2_list.append((v2))
        return self.k2_arr, self.v2_arr,self.v2_list

# Example usage
dict_reader = DictReader()
k2_arr1,v2_arr1,v2_list1 = dict_reader.read_dict(dic2)
 #**** create dictionary from keys(list) and multiple values(tuple) , using loop
k3 = [1,2]
v3 = ['a','b','c','d']
def creat_dict(k,v):
    lbl_dict =  dict()
    cnt = 0
    for i in k:
        lbl_dict.update({i: tuple([v[cnt],v[cnt+ 1]])})
        cnt = cnt + 2
        #lbl_dict.update({k[i]: tuple(v[i])})
    return lbl_dict
dic4 = creat_dict(k3,v3)
print(dic4)

# recerate above using zip: difference is each value is list then tuple
# Create pairs of values from v3 for each key in k3
def creat_dict1(k,v):
    # create list of 2 elemnets each
    pairs = [v[i:i+2] for i in range(0, len(v), 2)]
    # Create a dictionary by pairing k with pairs
    my_dict = dict(zip(k3, pairs))
    return my_dict
dic5 = creat_dict1(k3,v3)
print(dic5)

# *******check if current path exists,else create directory: os.path.exists(),os.makedirs()*********
pth = 'C://ML//env//tf//test_numpy'
if not os.path.exists(pth):
    os.makedirs(pth)
#********** get most nested folder from path : os.path.basename(), path upto last folder:os.path.dirname()   *****************
pth_folder_nm = os.path.basename(pth)
pth_bs = os.path.dirname(pth)
pth_bs_oneup = os.path.dirname(os.path.dirname(pth))
# better method 2: use pathlib library:  use Path().parents[i]
from pathlib import Path
pth_bs_v2 = Path(pth).parents[0]
pth_bs_oneup_v2 = Path(pth).parents[1]

# join 2 paths **** os.path.join(pth1,pth2)*****************

# ****create list of all files matching extension in  in a folder: use glob.glob(folder_pth + "//*.file_ext") ***
import glob
pth1 = os.getcwd()
py_files = glob.glob(pth1 + "//*.py")


# list all subdir in current dir: os.listdir()
dirs = os.listdir(pth)

# ***********os.walk(pth)  **** gives generator , with 3 tuples back
#   ##

for pth_upto_subdir, subdir_nm, files in os.walk(pth, topdown=False):
    print(pth_upto_subdir)
    print(subdir_nm)
    print(files)
# C://ML//env//tf//test_numpy\f1
# []
# ['areds_v1.py', 'speed_challenge_v1.py']
# C://ML//env//tf//test_numpy\f2
# []
# ['areds_v2.py', 'speed_challenge_v2.py']
# C://ML//env//tf//test_numpy\f3
# []
# ['areds_v3.py', 'speed_challenge_v3.py']
# C://ML//env//tf//test_numpy
# ['f1', 'f2', 'f3']
# []

 # ****** create list of all files matching extension in a folder and its subdirectories ****
path_list_files = []
for path2, subdirs1_categ, files1 in os.walk(pth):
    if len(files1) != 0:
        path_list_files = path_list_files + files1

print('****************')
# bubbsort()
bb1= np.random.randint(10,20,10)
def swap(x,y):
    temp = x
    y = x
    x= temp
    return x,y
def bbsort(x):
    for i in range(np.size(x)-1,0,-1):
        sorted = 1
        for j in range(i):
            sorted = 0
            if (x[j] > x[j+1]):
                 x[j],x[j+1] = swap(x[j],x[j+1])
                 sorted = 0

bbsort(bb1)


#********** slection sort ******
predicted = np.array([4, 25,  0.75, 11])
observed  = np.array([3, 21, -1.25, 13])
diff = predicted - observed
r1 = np.std(diff)
r2 =  r1*r1

# from sklearn.metrics import mean_squared_error
# from math import sqrt
#
# rms1 = sqrt(mean_squared_error(observed, predicted))
# print('h')

print("*************some interview questions***")
#onvert a given string int to int using a single line of code.
x = "123"
print(int(x))
# convert each char of string to list
y = []
[y.append(x[i]) for i in range(len(x))]
print(y)
x = "hello world"
# convert string to list of each word
y = x.split(" ")

# reverse a string.
x = "Abcde"
yRev = x[::-1]
yRev1 = []
# M1extract each element to list
for i in range(len(x)):
    yRev1.append(x[len(x)-1 -i])
# list to string
yRev1 = ''.join(yRev1)

# append directly [newElem overall]
yRev2 = ''
for char in x:
    yRev2 = char + yRev2
z=1
# minimum value in dictionary
dict_[min(dict_.keys(), key=(lambda k: dict_[k]))]