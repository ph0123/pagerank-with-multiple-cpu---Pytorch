
import time  # good version for
import numpy as np
import socket
import cpuinfo
import time
import datetime
import sys
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.distributed.rpc as rpc
import numpy as np

import time  # good version for
import numpy as np
import socket
import cpuinfo
import time
import datetime
import sys

# <int, int> map_list in python I use dictionary, map from index to object Vertex.
# define INITIAL_RESERVE_SIZE_MAX_EDGES 50
# define SEP ','
# define EXPORT_RESULT false
# define DEBUG_DETAILED false
INITIAL_RESERVE_SIZE_MAX_EDGES = 50
SEP = '\t'
EXPORT_RESULT = False
DEBUG_DETAILED = False
DEBUG = False

# <int, int> map_list in python I use dictionary, map from index to object Vertex.
# define INITIAL_RESERVE_SIZE_MAX_EDGES 50
# define SEP ','
# define EXPORT_RESULT false
# define DEBUG_DETAILED false
INITIAL_RESERVE_SIZE_MAX_EDGES = 50
SEP = ','
EXPORT_RESULT = False
DEBUG_DETAILED = False
DEBUG = False

_local_dict = {}



"""RPC with Torch"""  # version sum some values from their key. get values from all workers (including current rank)
"""run.py:"""




def intersect_dict(dict1):
    ret = {}
    for key in _local_dict:
        if key in dict1:
            ret[key] = dict1[key]
    return ret


def intersect_dict2(target_keys):
    res = 0.0
    for key in target_keys:
        _key = int(key)
        if _key in _local_dict:
            res += _local_dict[_key]
    return res


def run(dst, dict1):
    intersect = rpc.rpc_sync(dst, intersect_dict, args=(_local_dict,))
    return intersect


def run2(dst, target_keys):
    intersect = rpc.rpc_sync(dst, intersect_dict2, args=(target_keys,))
    return intersect


def create_dict(rank, size):
    local_dict = {}
    for i in range(0, 10):
        if (i % size == rank):
            local_dict[i] = 1.2 * (rank + 1)
    return local_dict


def print_hello(results, current_rank, rank):
    print(results, ' on rank ', rank, " from rank ", current_rank)

 #FUNCTION OF PAGERANK
def get_worker_hello():
    hostname = socket.gethostname()
    cpu = cpuinfo.get_cpu_info()['brand']
    s = "Node: " + hostname + ".\n" + "CPU: " + cpu + "."
    return s

def getFileName(filePath, withExtension=True, seperator='/'):
    dotPos = filePath.rfind('.')
    sepPos = filePath.rfind(seperator)
    if (sepPos != -1 and dotPos != -1 and withExtension == True):
        return filePath[sepPos + 1:]
    if (withExtension == False):
        return filePath[sepPos + 1:]
    return filename

# Print worker information
def get_worker_hello():
    hostname = socket.gethostname()
    cpu = cpuinfo.get_cpu_info()['brand']
    s = "Node: " + hostname + ".\n" + "CPU: " + cpu + "."
    return s

class Vertex:
    def __init__(self, index):
        self.vertexId = index
        self.links = []
        self.in_links = []
        self.pr = 0.15
        self.new_pr = 0.0

    def add_link(new_link):
        self.links.append(new_link)

    def on_push_recv(val):
        self.new_pr += val

    def finishing_work():
        self.pr = self.new_pr
        self.new_pr = 0.0

    def __str__(self):
        return "ID" + str(self.vertexId) + " " + str(self.pr) + " from object Vertex"

def load_data(data_file,my_rank, world_size):
    total_vertices = 0
    total_edges = 0
    total_rows = 0
    rows_skipped = 0
    # return all value without data_file
    isRankRoot = (my_rank==0)
    # bool isRankRoot = upcxx::rank_me() == 0
    rank_me = my_rank  # saddlebags::rank_me()
    # int rank_n = 1#upcxx::rank_n()
    line = ""
    vertex_str = ""
    # int prog_counter = 0
    format_neighbor_start_index = 2
    neighbor = -1
    DEBUG = False
    start_time = time.perf_counter()

    if (isRankRoot == True) and (DEBUG == True):
        print("Loading data from: " + str(data_file))

    if (rank == 0):
        print("loading data ...")
    file1 = open(data_file, 'r')
    Lines = file1.readlines()
    for line in Lines:
        if line == "\n":  # empty line
            rows_skipped += 1
            continue
        tokens = line.split(" ")

        if (len(tokens) < 2):
            print("[Rank " + str(rank_me) + "] ERROR: Unable to parse vertex for: " + str(line))
            rows_skipped += 1
            continue
        # Possible graph file formats like: source num_neighbors neighbor_1 neighbor_2 ... neighbor_n
        if (len(tokens) >= 3):
            format_neighbor_start_index = 2
        else:
            format_neighbor_start_index = 1
        vertex = int(tokens[0])
        total_rows += 1
        is_my_obj = ((vertex%world_size) == rank_me)
        total_edges += len(tokens)-2
        if (is_my_obj == True):
            new_obj = Vertex(vertex)
            #total_vertices += 1
            for i in range(format_neighbor_start_index, len(tokens)):
                neighbor = int(tokens[i])
                new_obj.links.append(neighbor)
            _local_dict[vertex] = new_obj
        if (total_vertices == 1 and DEBUG_DETAILED == True):
            print("[Rank " + str(rank_me) + "] Inserted first vertex <" + str(vertex) + ">.")
        # if(total_rows%100000==0) cout<<total_rows<<endl
        #total_edges = len(Lines)-rows_skipped
    file1.close()



    # for it in _local_dict:
    #    print(str(len(_local_dict[it].in_links)))
    if (DEBUG_DETAILED == True):
        timenow = datetime.datetime.now()
        print("[Rank " + str(rank_me) + "] "
              + str(timenow)
              + "Inserted objects: " + str(total_vertices)
              + " (Out of total objects: " + str(total_rows) + ")")

    end_data_gen = time.perf_counter()
    elapsed_time_data_gen = (end_data_gen - start_time) * 1000  # milisecond

    if rank ==0:
        timenow = datetime.datetime.now()
        print("[Rank ", rank_me, "] "
              , timenow
              , "Input file loaded with ", total_rows
              , " objects in ", elapsed_time_data_gen, " ms.")
    return total_vertices, total_edges, total_rows, rows_skipped


def get_max_pagerank(iter=0, detailed=True):
    s = ""
    max_pr = 0.0
    max_pr_id = 0
    # Iterate over all vertex objects
    for it in _local_dict:
        u = _local_dict[it]
        if (u.pr >= max_pr):
            max_pr = u.pr
            max_pr_id = u.vertexId
    if (detailed == True):
        s = s + "\n" + "[Iter " + str(iter) + "]" + " Max ID: " + str(max_pr_id) + ", Max PageRank:" + str(max_pr)
    else:
        s = s + str(max_pr) + str(SEP) + str(max_pr_id)
    return s

def get_max_pagerank_return(iter=0, detailed=True):
    s = ""
    max_pr = 0.0
    max_pr_id = 0
    # Iterate over all vertex objects
    for it in _local_dict:
        u = _local_dict[it]
        if (u.pr >= max_pr):
            max_pr = u.pr
            max_pr_id = u.vertexId
    return max_pr,max_pr_id

class Edge:
    def __init__(self, source, dest):
        self.source= source
        self.dest = dest



def add_outlinks(arr, source):
    for dest in arr:
        dest = int(dest)
        if dest in _local_dict:
            _local_dict[dest].in_links.append(int(source))

def Compute_Neighbor(arr):
    pr_neighbor = 0.0
    for i in arr:
        pr_neighbor += _local_dict[i].pr / len(_local_dict[i].links)
    return pr_neighbor


def run_iterations(rank, size):
    for element in _local_dict:
        if len(_local_dict) == 0:
            continue
        array_rpc = list(range(0, size))
        array_rpc.remove(rank)
        arr_send=[]
        for i in range(0, size):
            temp = []
            arr_send.append(temp)

        pr_neighbor = 0.0
        for in_vertex in _local_dict[element].in_links:
            arr_send[int(in_vertex)%size].append(in_vertex)
        #print(arr_send)
            # need to call rpc to dest worker to do it.  //parallel here
        futs = []
        for i in array_rpc:
            if len(arr_send[i]) > 0:
                my_target = "worker" + str(i)
                futs.append(rpc.rpc_async(my_target, Compute_Neighbor, args=(arr_send[i],)))
        pr_neighbor += Compute_Neighbor(arr_send[rank])

        for fut in futs:
            pr_neighbor = pr_neighbor+ fut.wait()

        #pr_neighbor +=
        _local_dict[element].new_pr = 0.85 * pr_neighbor + 0.15

    #end-warm-up
    rpc.api._wait_all_workers()

    for element in _local_dict:
        _local_dict[element].pr = _local_dict[element].new_pr
        _local_dict[element].new_pr = 0.0

def pagerank(rank, size, iterations):
    # start warm-up
    #check here.....
    run_iterations(rank,size)
    #print("pagerank after run: ", get_max_pagerank(0, False))
    # end warm-up and run n iterations

    rpc.api._wait_all_workers()
    start_time = time.perf_counter()
    for iter in range(0, iterations):
        for element in _local_dict:
            if len(_local_dict) == 0:
                continue
            if (iter > 0):
                array_rpc = list(range(0, size))
                array_rpc.remove(rank)
                arr_send = []
                for i in range(0, size):
                    temp = []
                    arr_send.append(temp)

                pr_neighbor = 0.0
                for in_vertex in _local_dict[element].in_links:
                    arr_send[int(in_vertex) % size].append(in_vertex)
                futs = []
                for i in array_rpc:
                    if len(arr_send[i]) > 0:
                        my_target = "worker" + str(i)
                        futs.append(rpc.rpc_async(my_target, Compute_Neighbor, args=(arr_send[i],)))
                pr_neighbor += Compute_Neighbor(arr_send[rank])

                for fut in futs:
                    pr_neighbor = pr_neighbor + fut.wait()

                _local_dict[element].new_pr = 0.85 * pr_neighbor + 0.15

        rpc.api._wait_all_workers()

        if (iter > 0):
            for element in _local_dict:
                _local_dict[element].pr = _local_dict[element].new_pr
                _local_dict[element].new_pr = 0.0

        rpc.api._wait_all_workers()
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000


def init_process(rank, size, filename,master_add, master_port, iterations, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master_add
    os.environ['MASTER_PORT'] = master_port

    my_name = "worker" + str(rank)
    my_target = "worker" + str((rank + 1) % size)

    #LOAD DATA HERE
    data_file = filename
    if (rank == 0):
        print("data file: ",data_file)
        print("Number of Iteration: ",iterations)

    total_vertices = 0
    total_edges = 0
    total_rows = 0
    rows_skipped = 0

    #LOAD DATA
    total_vertices, total_edges, total_rows, rows_skipped = load_data(data_file,rank, size)
    #END LOAD DATA, IT WILL STORE TO _local_dict
    rpc.init_rpc(my_name, rank=rank, world_size=size)  # initial_rpc

    #CALL rpc TO OTHER RANKS

    # std::ostringstream s_in
    s_in = ""

    array_rpc = list(range(0, size))
    array_rpc.remove(rank)

    for it in _local_dict:
        arr_send = []
        for i in range(0, size):
            temp = []
            arr_send.append(temp)
        u = _local_dict[it]
        source = u.vertexId
        #s_in += "check inlinks" + str(u.vertexId)
        for i in u.links:
            dest = int(i)
            arr_send[int(dest) % size].append(dest)
            # need to call rpc to dest worker to do it.  //parallel here
        for i in array_rpc:
            my_target = "worker" + str(i)
            if len(arr_send[i])>0:
                rpc.rpc_async(my_target, add_outlinks, args=(arr_send[i],source))
        add_outlinks(arr_send[rank],source)

    rpc.api._wait_all_workers()
    '''
    Run pagerank
    '''
    if(rank == 0):
        print("Running ... ")
    running_time  = pagerank(rank,size, iterations)
    #print(get_max_pagerank(iterations,True))
    rpc.api._wait_all_workers()

    if rank == 0:
        Max_pr,MaxID= get_max_pagerank_return(iterations,True)
        futs = []
        for i in range(1,size):
            my_target = "worker" + str(i)
            futs.append(rpc.rpc_async(my_target, get_max_pagerank_return, args=(iterations,True)))
        for fut in futs:
            Max_pr1 = fut.wait()[0]
            MaxID1 = fut.wait()[1]
            if(Max_pr<Max_pr1):
                Max_pr = Max_pr1
                MaxID = MaxID1
        s = "[Iter " + str(iterations) + "]" + " Max ID: " + str(MaxID) + ", Max PageRank:" + str(Max_pr)
        print(s)
        print("Total time for ", iterations, "iterations: ", running_time, "(ms)")  # ms
        print("Done!")
        s = "SUCCESS: PageRank finished in time: " + str(running_time) + " milliseconds"+ " (" + str(running_time / (60 * 1000)) + " minutes)"
        s= s + ", Ranks: " + str(size)
        s = s + ", Total Objects: " + str(total_rows)
        print(s)
        s=  "benchmark,platform,processes,dataset,vertices,edges,iterations,"
        s = s + "processing time (ms),"
        s = s +  "rows (skipped)"
        print(s)
        print( "PageRank" , SEP
                  , "Pytorch" , SEP
                  , size , SEP
                  , getFileName(data_file), SEP
                  , total_rows , SEP
                  , total_edges , SEP
                  , iterations , SEP
                  , running_time , SEP

                  , rows_skipped)
    rpc.shutdown()  # end_rpc



if __name__ == "__main__":
    print('Number of arguments:', len(sys.argv), 'arguments.')
    #print('Argument List:', str(sys.argv[0]))
    if len(sys.argv)!=6:
        print("You need 6 arguments")
        print("Syntax error <program> <input_file> <master_address> <master_port> <Number of workers> <number of iterations>")
    else:
        try:
            filename = sys.argv[1]
            master_add = sys.argv[2]
            master_port = sys.argv[3]
            size = int(sys.argv[4])
            iterations = int(sys.argv[5])

            processes = []
            for rank in range(size):
                p = Process(target=init_process, args=(rank, size,filename,master_add, master_port,iterations))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        except ValueError:
            print("Syntax error <program> <input_file> <master_address> <master_port> <Number of workers> <number of iterations>")
            print("Please check type of input parameter")
            print("example: simple_graph.txt localhost 29500 2 10")
