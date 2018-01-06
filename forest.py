
# coding: utf-8

# In[1]:

import random
import hashlib

f=()
hashValMap = dict()
curr_prime = 5
previous_hash_value = 0
first_char = ''


def rabin_karp_hash(P):
    def hash(s):
        current_hash_value = 0
        char_num = 0
        global previous_hash_value, first_char

        #if previous_hash_value == 0:
        for ch in s:
            current_hash_value += (ord(ch)-ord("A")+1)*(P**(len(s)-1-char_num))
            char_num += 1
        '''else:
            new_char_offset = (ord(s[len(s)-1])-ord("A")+1)
            temp1 = (previous_hash_value - ((ord(first_char)-ord("A")+1)*(P**(len(s)-1-char_num))))
            current_hash_value = (temp1 *(P) + new_char_offset) '''
        current_hash_value = current_hash_value % 100123456789
        #previous_hash_value = current_hash_value
        #first_char = s[len(s) - 1]
        return current_hash_value
    return hash;

def rabinHash(r, P):
        def hash(s):
            hashVal = 0 
            sizeWord = len(s);
            

            for i in range(0, sizeWord):
                #ord maps the character to a number
                #subtract out the ASCII value of "a" to start the indexing at zero
                hashVal += (ord(s[i]) - ord("A")+1)*(r**(sizeWord - i -1))
            return hashVal%P
        return hash

# Calculates a distinct hash function for a given string. Each value of the
# integer d results in a different hash value.
def hashf( d, st ):
    if d == 0: d = 0x01000193
    s =str(st)
    # Use the FNV algorithm from http://isthe.com/chongo/tech/comp/fnv/ 
    for c in s:
        d = ( (d * 0x01000193) ^ ord(c) ) & 0xffffffff;

    return d

# Computes a minimal perfect hash table using the given python list. It
# returns a tuple (G, V). G and V are both arrays. G contains the intermediate
# table of values needed to compute the index of the value in V. V contains the
# values of the dictionary.
def CreateMinimalPerfectHash(index, dict ):
    
    #print(dict)
    size = len(dict)

    # Step 1: Place all of the keys into buckets
    buckets = [ [] for i in range(size) ]
    G = [0] * size
    values = [None] * size
    
    for key in dict.keys():
        # print(dict[key],"  hash:",key)
        buckets[hashf(0,key) % size].append( key )

    # Step 2: Sort the buckets and process the ones with the most items first.
    buckets.sort( key=len, reverse=True )        
    for b in range( size ):
        bucket = buckets[b]
        # print(bucket)
        if len(bucket) <= 1: break
        
        d = 1
        item = 0
        slots = []
        maxIterations = size;

        # Repeatedly try different values of d until we find a hash function
        # that places all items in the bucket into free slots
        while item < len(bucket): #and maxIterations>0:
            slot = hashf(d,bucket[item] ) % size
            if values[slot] != None or slot in slots:
                d += 1
                item = 0
                slots = []
            else:
                slots.append( slot )
                item += 1
            #maxIterations-=1;
        #print(slots)
        #if maxIterations==0:
            #return None
        
        G[hashf(0,bucket[0]) % size] = d
        for i in range(len(bucket)):
            values[slots[i]] = dict[bucket[i]]
            
    # Only buckets with 1 item remain. Process them more quickly by directly
    # placing them into a free slot. Use a negative value of d to indicate
    # this.
    freelist = []
    for i in range(size): 
        if values[i] == None: freelist.append( i )

    for b in range( b, size ):
        bucket = buckets[b]
        if len(bucket) == 0: break
        slot = freelist.pop()
        # We subtract one to ensure it's negative even if the zeroeth slot was
        # used.
        G[hashf(0,bucket[0]) % size] = -slot-1 
        values[slot] = dict[bucket[0]]
        
        
    # print(G)
    # print(values)
    # Look up a value in the hash table, defined by G and V.
    def PerfectHash(kmer):
        s = int(hashlib.sha1(kmer.encode('utf-8')).hexdigest(), 32) % (10 ** index)
        d = G[hashf(0,s) % len(G)]
        if d < 0: return (-d-1)
        return (hashf(d,s) % len(values))
    return PerfectHash     

def largestPrime(n):
    i=0
    j=0
    winner=-1
    primes=[True]*n
    primes[0]=primes[1]=False
    for i in range(2,n):
        if primes[i]:  
            winner = i
            for j in range(i+i,n,i):
                primes[j]=False
    return winner

def newRabinKarp(S, k):
    global previous_hash_value, first_char
    global curr_prime
    #R = max(4, k*(len(S)**2))
    P = 100123456789#find_next_prime(curr_prime)#100123456789#largestPrime(R)
    
    r = find_next_prime(curr_prime)#random.randint(k,P-1)
    curr_prime = r
    previous_hash_value = ''
    first_char = 0
    f = rabinHash(r,P)
    #print(r,"  ",P)
    while not isInjective(f,S):
        #print(r," ",P)
        previous_hash_value = ''
        first_char = 0
        #P = find_next_prime(curr_prime)
        r = find_next_prime(curr_prime)
        curr_prime = r
        f = rabinHash(r,P)
    
    return f;

def find_next_prime(n):
    return find_prime_in_range(n, 2*n)

def find_prime_in_range(a, b):
    for p in range(a, b):
        for i in range(2, p):
            if p % i == 0:
                break
        else:
            return p
    return None

def createHashValMap(f, S):
    global previous_hash_value, first_char
    previous_hash_value = 0
    first_char = ''
    global hashValMap
    i=0
    for s in S:
        hashValMap[f(s)] = i
        i+=1
    def PerfectHash(s):
        global previous_hash_value, first_char
        previous_hash_value = 0
        first_char = ''
        if f(s) in hashValMap:
            return hashValMap[f(s)]
        else:
            return 0
    return PerfectHash

    
def isInjective(index, S):
    global previous_hash_value, first_char
    
    hashValSet = dict();
    for s in S:
        hashValSet[int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 32) % (10 ** index)] = s;

    if(len(hashValSet)==len(S)):
        return hashValSet;
    else:
        return None;
    
def GenerateHash(S, k):
    global f
    '''f = newRabinKarp(S, k)
    #print(isInjective(f,S))
    # print(r,"  ",P)
    while not isInjective(f,S):
        f = newRabinKarp(S, k)'''

    index =14
    rkHashValues = None
    rkHashValues=isInjective(index,S)
    while rkHashValues == None:
        index+=1
        rkHashValues=isInjective(index,S)
    
    g = CreateMinimalPerfectHash(index,rkHashValues)#createHashValMap(f, S)
    print("Hash Function created")
    return g

def getKMer(str,k):
    if(len(str)<k):
        return []
    curKMer = ''
    for index in range(k):
        curKMer = curKMer + str[index]
        
    kMerlist = [curKMer];
    
    for index in range(k,len(str)):
        curKMer = curKMer[1:]
        curKMer = curKMer + str[index]
        kMerlist.append(curKMer)
        
    
    return kMerlist

def getInOutMatrices(kMerList, hashFunc):
    inMatrix = {}
    outMatrix = {}
    mapping = {}
    
    if(len(kMerList) == 0):
        return {"IN":{},"OUT":{}, "MAPPING":{}}
    
    mapping[hashFunc(kMerList[0])] = kMerList[0]
    mapping[hashFunc(kMerList[len(kMerList)-1])] = kMerList[len(kMerList)-1]
    
    inMatrix[hashFunc(kMerList[0])] = [False,False,False,False]
    outMatrix[hashFunc(kMerList[len(kMerList)-1])] = [False,False,False,False]
    for index in range(1,len(kMerList)):
        idx = getBPIndex(kMerList[index-1][:1])
        
        #apply hash function on curKMer
        curKMer = hashFunc(kMerList[index])
        mapping[curKMer] = kMerList[index]
        if(curKMer not in inMatrix):
            inMatrix[curKMer] = [False,False,False,False]
        
        inMatrix[curKMer][idx] = True
        
        #apply hash function on curOutKMer
        curOutKMer = hashFunc(kMerList[index-1])
        mapping[curOutKMer] = kMerList[index-1]
        
        idx = getBPIndex(kMerList[index][-1:])
        
        if(curOutKMer not in outMatrix):
            outMatrix[curOutKMer] = [False,False,False,False]
            
        outMatrix[curOutKMer][idx] = True
        
    return {"IN":inMatrix,"OUT":outMatrix, "MAPPING":mapping}


class TreeNode:
     def __init__(self, x, parent, height):
        self.val = x   
        self.height = height
        self.parent = parent
        

def mergeMatrix(inMatrix, outMatrix):
    
    for key, value in outMatrix.items():
        curInList = inMatrix[key]
        curOutList = outMatrix[key]
        for i, val in enumerate(curOutList):
            curOutList[i] = curOutList[i] | curInList[i]
    
    return outMatrix

def getBPIndex(bp):
    dict = {'A':0,'C':1,'G':2,'T':3}
    retIndex = 0
    if(bp in dict):
        retIndex = dict[bp]
    return retIndex

def getBPFromIndex(idx):
    #dict = {'A':0,'C':1,'G':2,'T':3}
    lst = ['A','C','G','T']
    if(idx>len(lst)): 
        return ''
    return lst[idx]

def getNeighbours(kMerHash, inMatrix, outMatrix, mapping, hashFunc):
    neighbours = []
    kMer = mapping[kMerHash]
    if(kMer is None):
        return neighbours
    
    ##from out matrix
    if(kMerHash in outMatrix):
        adjList = outMatrix[kMerHash]
        for index in range(0,len(adjList)):
            if(adjList[index] == True):
                curMer = kMer[1:] + getBPFromIndex(index)
                neighbours.append((getBPFromIndex(index), curMer))
    
    ##from in matrix
    '''if(kMerHash in inMatrix):
        adjList = inMatrix[kMerHash]
        for index in range(0,len(adjList)):
            if(adjList[index] == True):
                curMer = getBPFromIndex(index) + kMer[:-1] 
                neighbours.append(curMer)'''
            
    return neighbours

def generateForest(inMatrix, outMatrix, mapping, hashFunc, maxHeight):
    forestRoots = []
    visited = set()
    hashNodeMap = {}
    #iterate over mapping to create trees
    for key, value in mapping.items():
        if(key not in visited):
            treeRoot = generateTree(value, visited,inMatrix, outMatrix, mapping, maxHeight, hashFunc, hashNodeMap)
            forestRoots.append(treeRoot)
    #return hashNodeMap
    print("rootcount ::" + str(len(forestRoots)))
    return hashNodeMap
    

def generateTree(rootKMer, visited,inMatrix, outMatrix, mapping, maxHeight, hashFunc, hashNodeMap):
    
    root = TreeNode(hashFunc(rootKMer),None,1)
    hashNodeMap[hashFunc(rootKMer)] = root
    visited.add(hashFunc(rootKMer))
    
    lst = []
    lst.append(hashFunc(rootKMer))
    while(maxHeight>0):
        maxHeight = maxHeight - 1
        
        newlst = []
        for index in range(0,len(lst)):
            parent = lst[index]
            parentNode = hashNodeMap[parent]
            #print("looking for:: " + str(parent.val))
            #kMerHash = hashFunc(mapping[parent.val])
            neighbours = getNeighbours(parent, inMatrix, outMatrix, mapping, hashFunc)
            #if(parent == hashFunc("GACT")):
                #print("neighbours :: ")
                #print(neighbours)
            #add to visited
            for ni in range(0,len(neighbours)):
                curnode = neighbours[ni]
                curnodeHash = hashFunc(curnode[1])
                if(curnodeHash not in visited):
                    visited.add(curnodeHash)
                    #print(curnode[0])
                    newnode = TreeNode(curnode[0], parentNode, 1)
                    updateHeight(newnode, 1)
                    #print(curHeight)
                    hashNodeMap[curnodeHash] = newnode
                    newlst.append(curnodeHash)
            
        lst = newlst
        
    #print(rootKMer)    
    root.val = rootKMer
   
    return root
#update height while forest creation
def updateHeight(node,curHeight):
    
    curNode = node.parent
    height = curHeight + 1
    while(curNode is not None):
        if(curNode.height > height):
            break;
        
        curNode.height = height
        curNode = curNode.parent
        height = height+1
    
    return

def queryMember(kMerString, hashNodeMap, hashFunc, outMatrix):
    leftKMer = kMerString[:-1]
    rightKMer = kMerString[1:]
    
    leftKMerHash = hashFunc(leftKMer)
    #for index in range(0,len(kMerList)):
     #   curKmer = kMerList[index]
        #ret = isMember()
    
    if(isMember(leftKMer,hashNodeMap, hashFunc) and isMember(rightKMer,hashNodeMap, hashFunc)):
        #check in inmatrix
        return outMatrix[leftKMerHash][getBPIndex(rightKMer[-1:])]
    else:
        return False

def isMember(kMerString, hashNodeMap, hashFunc):
    kMerHash = hashFunc(kMerString)
    #print("looking for:" + str(kMerHash))
    if(kMerHash not in hashNodeMap):
        return False
    
    node = hashNodeMap[kMerHash]
    #if(node.parent is None):
    #    print("root")
        #check in in matrix
    #   return kMerString[1:] == node.val and inMatrix[kMerHash][getBPIndex(kMerString[-1:])]
    #else:
    while(node.parent is not None and len(kMerString)>0):
        #print("current::"+kMerString[-1:] + ":: expected::" + node.val)
        if(node.val != kMerString[-1:]):
            #print("character mismatch while ascending::"+kMerString[-1:] + ":: expected::" + node.val)
            return False
        kMerString = kMerString[:-1]
        node = node.parent
      
    #whole kmerstring consumed
    if(len(kMerString) == 0):
        return True
    else:
        if(node.val[-1*len(kMerString):] != kMerString):
            #print("last length not matching:: expected::" + node.val[-1*len(kMerString):] + " :: current::: " +kMerString)
            return False
    
    #compare and return val
    return True



def getStringTillRoot(initialStr,node,K):
    
    if(node is None):
        return initialStr
    
    while(node.parent is not None):
        initialStr = node.val + initialStr
        node = node.parent
    
    wholeStr = node.val
    
    c = len(wholeStr) - len(initialStr)
    
    if(c>0):
        finalStr = wholeStr[-c:] + initialStr 
    else:
        finalStr = initialStr[-K:]
    
    return finalStr
    
def getNodeToAttach(childHash,childStr, inMatrix, hashFunc, hashNodeMap):
    lst = inMatrix[childHash]
    for idx in range(0,len(lst)):
        if(lst[idx] == True):
            firstChar = getBPFromIndex(idx)
            finalStr = firstChar+childStr[:-1]
            hashVal = hashFunc(finalStr)
            return hashNodeMap[hashVal]
    
    return None

def getChildren(node, nodeStr, hashFunc, outMatrix, hashNodeMap):
    
    nodeHash = hashFunc(nodeStr)
    lst = outMatrix[nodeHash]
    children = []
    for idx in range(0,len(lst)):
        if(lst[idx] == True):
            childStr = nodeStr[1:] + getBPFromIndex(idx)
            childHash = hashFunc(childStr)
            childNode = hashNodeMap[childHash]
            if(childNode.parent == node):
                children.append(node)
            
    return children

def getParentStr(node, nodeStr, hashFunc, hashNodeMap, inMatrix):
    
    idx = getBPIndex(nodeStr[-1:])
    hashVal = hashFunc(nodeStr)
    l1 = inMatrix[hashVal]
    print(l1)
    lst = []
    for i in range(0,len(l1)):
        if(l1[i] == True):
            lst.append(getBPFromIndex(i))
            
    #lst = ['A','C','G','T']
    
    for i in range(0,len(lst)):
        kMerStr = lst[i] +  nodeStr[:-1]
        print(kMerStr)
        print(kMerHash)
        kMerHash = hashFunc(kMerStr)
        kMerNode = hashNodeMap[kMerHash]
        
        if(kMerNode == node.parent):
            print("found::" + kMerStr)
            return kMerStr
    
    return None
#can improve by adding break 
def alterParentComponentHeight(node, nodeStr, hashFunc, inMatrix, outMatrix, hashNodeMap):
    
    node.height = 1
    
    while(node is not None):
        #print("par::" + nodeStr)
        children = getChildren(node, nodeStr, hashFunc, outMatrix, hashNodeMap)
        maxHeight = 1
        
        for i in range(0,len(children)):
            maxHeight = max(maxHeight, children[i].height + 1)
        
        #if(node.height  maxHeight):
         #   break;
        #else:
        node.height = maxHeight
        if(node.parent is not None):
            nodeStr = getParentStr(node, nodeStr, hashFunc, hashNodeMap, inMatrix)
            #print("from getparetbtstr::" + nodeStr)
        
        node = node.parent
            
    return

#update when attaching
#maxheight = 3KLogSig
def alterHeight(nodeChild, childStr, maxHeight, K, hashFunc, inMatrix, outMatrix, hashNodeMap):
    curNode = nodeChild.parent
    curHeight = nodeChild.height + 1 
    while(curNode.parent is not None):
        
        
        if(curNode.height > curHeight):
            break;
        curNode.height = curHeight
        curHeight = curHeight + 1
        #height greater then 3klogsig
        if(curHeight>maxHeight):
            if(curNode.parent is not None):
                #detach tree 1. parent component update height...  
                #2. child component update root string
                #curNode.val = getStringTillRoot(curNode.val,curNode.parent,K)
                parentNode = curNode.parent
                parentStr = getStringTillRoot(parentNode.val,parentNode.parent,K)
                curNode.val = parentStr[1:] + curNode.val
                curNode.parent = None
                alterParentComponentHeight(parentNode, parentStr, hashFunc, inMatrix, outMatrix, hashNodeMap)
                return                
                
        curNode = curNode.parent
        
def insertEdge(fromStr, toStr, hashFunc ,hashNodeMap, inMatrix, outMatrix, K):
    outCharIndex = getBPIndex(toStr[-1:]) 
    inCharIndex = getBPIndex(fromStr[:1])
    toHash = hashFunc(toStr)
    fromHash = hashFunc(fromStr)
    if(outMatrix[fromHash] is not None):
        outMatrix[fromHash][outCharIndex] = True
        if(inMatrix[toHash] is not None):
            inMatrix[toHash][inCharIndex] = True
        
    return
        
def deleteEdge(n1Str, n2Str, hashFunc ,hashNodeMap, inMatrix, outMatrix, K):
    if((n1Str[1:] + n2Str[-1:]) == n2Str):
        #correct
        temp = n1Str
    elif((n2Str[1:] + n1Str[-1:]) == n1Str):
        #swap
        temp = n1Str
        n1Str = n2Str
        n2Str = temp
    else:
        print("no edge exist")
        return
    
    K1Height = round(math.log10(4)*K)
    K2Height = round(math.log10(4)*K*2)
    K3Height = round(math.log10(4)*K*3)
    
    outChar = n2Str[-1:]
    
    n1Hash = hashFunc(n1Str)
    n2Hash = hashFunc(n2Str)
    
    #remove edge from in and out matrix
    charVal = n2Str[-1:]
    #print("removing::" + n1Str + " :: " + str(n1Hash) + " :: " + charVal )
    outMatrix[n1Hash][getBPIndex(charVal)] = False
    inMatrix[n2Hash][getBPIndex(n1Str[:1])] = False
    
    
    node1 = hashNodeMap[n1Hash]
    node2 = hashNodeMap[n2Hash]
    if(node1 is None or node2 is None):
        return
    
    #check if edge exists in forest
    if(node1.parent != node2 and node2.parent != node1):
        return
    
    if(node1.parent == node2):
        nodeParent = node2
        nodeChild = node1
        nodeChildStr = n1Str
        nodeParentStr = n2Str
        nodeParentHash = n2Hash
    else:
        nodeParent = node1
        nodeChild = node2
        nodeChildStr = n2Str
        nodeParentStr = n1Str
        nodeChildHash = n1Hash
      
    #delete forest edge
    nodeChild.parent = None
    
    #child component processing
    #case 1: child component has enough height => find string of root node
    #case 2: child component has lesser height than KlogSig => find a tree to attach this using IN matrix then process new tree
        # to check if height is less than 3klogsig else break that
        # if no tree found to attach leave it
        
    #case1 impl
    if(nodeChild.height > K1Height):
        #find string
        nodeChild.val = getStringTillRoot(nodeChild.val,nodeParent,K)
        return
    #case 2
    else:
        #find tree to attach using IN matrix
        otherTreeNode = getNodeToAttach(nodeChildHash,nodeChildStr, inMatrix, hashFunc, hashNodeMap)
        if(otherTreeNode is not None):
            nodeChild.parent = otherTreeNode
            alterHeight(nodeChild, nodeChildStr, K3Height, K, hashFunc, inMatrix, outMatrix, hashNodeMap)
        else:
            nodeChild.val = nodeChildStr
    
    #parent component processing
    #print(nodeParentStr)
    alterParentComponentHeight(nodeParent, nodeParentStr, hashFunc, inMatrix, outMatrix, hashNodeMap)
    
    return
    


# In[5]:

import time
import math
#import kMerList
#from kMerList import getKMer
#from kMerList import getInOutMatrices

def readFastaFile(filename):
    file = open(filename,"r")
    dnaSeq = ""
    curLine = file.readline()
    print(curLine)
    while(curLine): 
        #print(curLine)
        if(">" not in curLine):
            dnaSeq = dnaSeq + curLine
        #else:
            #print(curLine)
        curLine = file.readline()
    
    return dnaSeq

def readFastqFile(filename):
    file = open(filename,"r")
    dnaSeq = ""
    curLine = file.readline()
    print(curLine)
    while(curLine): 
        #print(curLine)
        if(curLine[:1] in ['A','C','G','T']):
            dnaSeq = dnaSeq + curLine.strip()
        #else:
            #print(curLine)
        curLine = file.readline()
    print("Sequence length : " + str(len(dnaSeq)))
    return dnaSeq  

# compiled function to generate De Bruijn Grpah using IN, OUT matrices
def generateDeBruijnGraph(filename, k):
    file = open(filename,"r")
    queryStr = "ACCTGACAGTGCGGGCTTTTTTT"
    retVal = {}
    #get the first read
    dnaSeq = readFastqFile(filename)
    print("File read complete")
    #dnaSeq = "ACGTAGGACGTAGAA"
    print("Generating kmers...")
    kmerlist = getKMer(dnaSeq, k)
    print("kmer list generated")
    #print(kmerlist)
    start_time = time.time()
    setList = set(kmerlist)
    print("Generating Hash Function...")
    hashfunc = GenerateHash(setList,k)
    
    print("Hash function generation time: %s seconds ---" % (time.time() - start_time))
    
    print("Generating IN OUT Matrix...")
    start_time = time.time()
    matrixDict = getInOutMatrices(kmerlist, hashfunc)
    print("IN OUT matrix generation time: %s seconds ---" % (time.time() - start_time))
    #print("IN matrix:",matrixDict["IN"])
    #print("OUT matrix:",matrixDict["OUT"])
    #print(matrixDict)
    #print(matrixDict["MAPPING"][3])
    #print(matrixDict["IN"][3])
    #print(matrixDict["OUT"][3])
    #print(getNeighbours(3, matrixDict["IN"], matrixDict["OUT"], matrixDict["MAPPING"], hashfunc))
    maxHeight = round(math.log10(4)*3*k)

    
    inMatrix = matrixDict["IN"]
    outMatrix = matrixDict["OUT"]
    retVal["IN_MATRIX"] = inMatrix
    retVal["OUT_MATRIX"] = outMatrix
    retVal["K"] = k
    
    print("Generating Forest...")
    
    start_time = time.time()
    
    hashNodeMap = generateForest(matrixDict["IN"], matrixDict["OUT"], matrixDict["MAPPING"], hashfunc, maxHeight)
    
    print("Forest generation time: %s seconds ---" % (time.time() - start_time))
    retVal["FOREST"] = hashNodeMap
    retVal["HASH_FUNC"] = hashfunc

    #FREE MEMORY
    #print(matrixDict["MAPPING"])
    matrixDict["MAPPING"] = None
    #print(isMember("GACTG",hashNodeMap,hashfunc,inMatrix))
    return retVal

def membershipQuery(queryStr,deBruijnData):
    inMatrix = deBruijnData["IN_MATRIX"]
    outMatrix = deBruijnData["OUT_MATRIX"]
    hashNodeMap = deBruijnData["FOREST"]
    hashfunc = deBruijnData["HASH_FUNC"]
    k = deBruijnData["K"]
    
    if(len(queryStr) != 33):
        print("Invalid query. Enter query string of length 33.")
        return
    
   
    start_time = time.time()
    ret = queryMember(queryStr, hashNodeMap, hashfunc, outMatrix)
    if(ret == True):
        print(queryStr + " exists in graph")
    else:
        print(queryStr + " does not exist in graph")
    print("Membership query time: %s seconds ---" % (time.time() - start_time))

def removeEdge(queryStr,deBruijnData):
    inMatrix = deBruijnData["IN_MATRIX"]
    outMatrix = deBruijnData["OUT_MATRIX"]
    hashNodeMap = deBruijnData["FOREST"]
    hashfunc = deBruijnData["HASH_FUNC"]
    k = deBruijnData["K"]
    
    if(len(queryStr) != 33):
        print("Invalid query. Enter query string of length 33.")
        return
    
    print("delete:: " + queryStr)
    #print(inMatrix)
    #print(outMatrix)
    start_time = time.time()
    deleteEdge(queryStr[:-1], queryStr[1:], hashfunc ,hashNodeMap, inMatrix, outMatrix, k)
    print("Deletion time: %s seconds ---" % (time.time() - start_time))

def addEdge(queryStr,deBruijnData):
    inMatrix = deBruijnData["IN_MATRIX"]
    outMatrix = deBruijnData["OUT_MATRIX"]
    hashNodeMap = deBruijnData["FOREST"]
    hashfunc = deBruijnData["HASH_FUNC"]
    k = deBruijnData["K"]
    
    if(len(queryStr) != 33):
        print("Invalid query. Enter query string of length 33.")
        return
    
    
    print("insert:: " + queryStr)
    #print(inMatrix)
    #print(outMatrix)
    start_time = time.time()
    insertEdge(queryStr[:-1], queryStr[1:], hashfunc ,hashNodeMap, inMatrix, outMatrix, k)
    print("Insertion time: %s seconds ---" % (time.time() - start_time))


# In[3]:

#"dataset.fna"
fileName = input("Enter full file path..\n")
deBruijnData = generateDeBruijnGraph(fileName,32)


# In[4]:

#queryStr = "GGTGGTCGGTTCGAAAAACTC"
userInput = input("\nMenu: \n Press 1 for Membership Query \n Press 2 to delete edge \n Press 3 to add edge \n Press 4 to quit\n")
#Enter query string... Type quit to stop::: ")
while(userInput != "4"):
    if(userInput == "1"):
        queryStr = input("Enter the query string of 33 characters: \n")
        membershipQuery(queryStr,deBruijnData)
    elif(userInput == "2"):
        queryStr = input("Enter the string of 33 characters to delete edge: \n")
        removeEdge(queryStr,deBruijnData)
    elif(userInput == "3"):
        queryStr = input("Enter the string of 33 characters to add edge: \n")
        addEdge(queryStr,deBruijnData)
    else:
        print("Invalid choice.. Try again ... \n")
    userInput = input("\nMenu: \n Press 1 for Membership Query \n Press 2 to delete edge \n Press 3 to add edge \n Press 4 to quit\n")

