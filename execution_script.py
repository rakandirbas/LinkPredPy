#!/home/rdirbas/mypy/bin/python
import os.path, sys
import time
import subprocess as sp
import os
from itertools import izip
import shutil

def main():
    
    args = sys.argv
    choices = []
    
    if len(args) == 5:
        dataset_id = int(args[1])
        if args[2] == '-from:to':
            exp_range = args[3]
            min_exp, max_exp = exp_range.split(":") 
            min_exp = int(min_exp)
            max_exp = int(max_exp)
            choices = range(min_exp, max_exp+1)
            
            if args[4] == 'y': 
                large_data_set = True
            else:
                large_data_set = False
            
        elif args[2] == '-list':
            the_exps_list = args[3]
            the_exps_list = the_exps_list.replace('[','').replace(']','').split(',')
            for item in the_exps_list:
                choices.append(int(item))
                
            if args[4] == 'y': 
                large_data_set = True
            else:
                large_data_set = False
            
    elif len(args) == 3:
        dataset_id = int(args[1])
        min_exp = 1
        max_exp = 27
        choices = range(min_exp, max_exp+1)
        
        if args[2] == 'y': 
            large_data_set = True
        else:
            large_data_set = False
    else:
        dump = "script datasetid [ {-from:to 1:27} | {-list [3,4,8]} ] y_or_n_if_large_set"
        raise Exception(dump)
        
    
    if dataset_id == 1:
        job_name = "caltech"
    elif dataset_id == 2:
        job_name = "fb333"
    elif dataset_id == 3:
        job_name = "fb150"
    elif dataset_id == 4:
        job_name ="fb1034"
    elif dataset_id == 5:
        job_name ="GP5"
    elif dataset_id == 6:
        job_name ="GP6"
    elif dataset_id == 7:
        job_name ="SNAP_FBComb"#was FB100Comb mistake
    elif dataset_id == 8:
        job_name ="Roch"
    elif dataset_id == 9:
        job_name ="Buck"
    elif dataset_id == 10:
        job_name ="fb747"
    elif dataset_id == 11:
        job_name ="fb786"
    elif dataset_id == 12:
        job_name ="Reed"
    elif dataset_id == 13:
        job_name ="fb224"
    else:
        raise Exception("Please specify a valid dataset id!!")
    
#     if large_data_set:
#         memory = 50000
#     else:
#         memory = 15000

    memory = 15000
    
    main_job_path = '/home/rdirbas/final_tests/' + str(dataset_id) + '/'
    
    if os.path.exists(main_job_path):
        print "Dataset folder already exists! Would you like me to overwrite it (y,n)? "
        user_input = raw_input()
        if user_input == 'y':
            shutil.rmtree(main_job_path)
            os.makedirs(main_job_path)
        elif user_input != 'n':
            raise Exception("Unknown command bro!")
        
    else:
        os.makedirs(main_job_path)
    
    job_paths = []
    
    for choice in choices:
        job_path = main_job_path + str(choice) + '/'
        
        if os.path.exists(job_path):
            print "Experiement folder already exists! Would you like me to overwrite it (y,n)? "
            user_input = raw_input()
            if user_input == 'y':
                shutil.rmtree(job_path)
                os.makedirs(job_path)
                job_paths.append(job_path)
            else:
                raise Exception("Don't know what to do then, so I have to stop!")
        else:
            os.makedirs(job_path)
            job_paths.append(job_path)
        
    for choice, job_path in izip(choices, job_paths):
        run_script(dataset_id = dataset_id, choice=choice, job_path = job_path, job_name=job_name, memory=memory, cpus=1, large_data_set=large_data_set)


def run_script(dataset_id, choice, job_path, job_name, memory=20000, cpus=1, large_data_set=False):
    """
    The error and output files will be stored like this:
    job_path+job_name+ '_' +choice.out
    
    The job name on qstat will be like this:
    job_name + '_' +choice
    
    Parameters:
    -----------
    dataset_id: the id of the dataset to use.
    choice: the exp_id.
    job_path: where to save the outputs.
    job_name: dataset name.
    memory: amount of memory in MB.
    cpus: the number of CPUs to use.
    
    """
    
    low_memory_choices = [1,2,3,4,5,18,19,23,24,25,26,27,31,32,33]
    midium_memory_choices = [16,17,20,21,22, 28, 30, 39]
    many_cpus = [6, 7, 29, 34, 35, 36]
    
    if choice in low_memory_choices:
        memory = 6000
    elif choice in midium_memory_choices:
        memory = 15000
        
    dataset_size_choice = None
    
    if large_data_set:
        dataset_size_choice = 'y'
    else:
        dataset_size_choice = 'n'   
        
    if choice in many_cpus:
        cpus = 5
    
    job_name = job_name
    memory = str(memory) + "M"
    processors = str(cpus)
    choice = str(choice)
    dataset_id = str(dataset_id)
    
    print 'Running on dataset %s the exp %s with RAM=%s MB and dataset large or not: %s' % (dataset_id, choice, memory, dataset_size_choice)
    
    command = "/home/rdirbas/mypy/bin/python /home/rdirbas/Graph/Graph/runs/_0.py " + dataset_id + " " + choice + " " + dataset_size_choice

    job_string = """#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -N %s
#$ -o %s.out
#$ -e %s.err
#$ -M gucko.gucko@gmail.com
#$ -m abe
#$ -l mem_free=%s
#$ -pe simple %s
%s""" % (job_name + '_' +choice, job_path+job_name+ '_' +choice,  job_path+job_name+ '_' +choice, memory, processors, command)

    p = sp.Popen("qsub", stdout=sp.PIPE, stdin=sp.PIPE, shell=True)
    (output, err) = p.communicate(job_string)
    print output ,"\n\n", err
    time.sleep(0.5)




main()







    