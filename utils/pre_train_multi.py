import sys
import os
import threading

def get_node_list(astr):
    a2 = astr[2:-1].split(',')
    nlist=[]
    for a in a2:
        if '-' not in a:
            nlist.append(a)
        else:
            nlist+=[str(i) for i in range(int(a.split('-')[0]),int(a.split('-')[1])+1)]
    return(nlist)



nodes = ['c'+n for n in get_node_list(os.environ['SLURM_NODELIST'])]
print('NODES',nodes)
first = True
node_string = ""
for node in nodes:
    if not first:
        node_string += ','
    node_string += '"' + node + ':12345"'
    first = False

for i, node in enumerate(nodes):
    config = ('TF_CONFIG='+'\'"\'"\'{"cluster": {"worker": ['+node_string+']}, "task": {"index": '+str(i)+', "type": "worker"}}\'"\'"\'')

    print(node)
    command = 'ssh '+node+" 'conda activate yourenv;"+" cd /path_to_code/ &&"+config+" python pretrain_contmask.py /path_to_code/configs/loc_span_config.json'"

    print(command)
    
    def thread_function(command):
        os.system(command)

    thread = threading.Thread(target=thread_function, args=(command,))


    thread.start()
thread.join()

