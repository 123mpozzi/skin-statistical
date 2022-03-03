import subprocess
import psutil
import click
from utils.db_utils import *

# Parallelize the predictions by calling predict.py multiple times

cmd_root = 'python main.py single '

# TODO: try work=-2
# TODO: multiprocess: do they touch the dat.csv? if so change it in the py

def determine_workers(workers: int) -> int:
    '''
    Determine the number of workers automatically based on
    the system resources
    '''
    # on auto, # workers = # physical cpu cores
    if workers == -1:
        workers = psutil.cpu_count(logical=False)
    elif workers == -2: # use all cores, even logical ones
        workers = psutil.cpu_count()
    return workers

def determine_workers_per_db(db_sizes: dict, workload: int, workers: int, debug: bool) -> dict:
    '''
    Try to assign a proper number of workers per each dataset

    eg: Workload = 300.
        : If db1 has 200  images  -> 1 worker
        : If db2 has 1000 images  -> 4 workers
    '''
    remaining_workers = workers
    
    while remaining_workers > 0:
        # copy because it is being edited in the loop
        for entry in db_sizes.copy():
            if debug:
                print(db_sizes)
                print(remaining_workers)
            # Just consider db entries, not workers-related ones
            if entry[-1] == 'w':
                continue

            key = entry+'w' # key containing db + number of workers (eg. ECUw : 1)

            if remaining_workers != 0:
                # db need more work than a workload unit
                if db_sizes[entry] > workload:
                    # Add a worker
                    if key in db_sizes:
                        db_sizes[key] = int(db_sizes[key]) + 1
                    else:
                        db_sizes[key] = 1
                    remaining_workers = remaining_workers -1
                # A workload unit is enough
                else:
                    # If all work needed in current db is already assigned, continue
                    if key in db_sizes and db_sizes[key] == 1:
                        continue

                    # Add a worker
                    db_sizes[key] = 1
                    remaining_workers = remaining_workers -1
            else:
                break
    return db_sizes

def calculate_workload(models: list, workers: int, use_only_test_set: bool = False) -> list:
    '''
    Calculate the workload by measuring the size of each db

    Return also a dict containing each db with their size
    '''
    # Get total db size
    db_sizes = {}
    total_db_size = 0
    for m in models:
        if use_only_test_set:
            image_paths = get_db_by_name(m).get_test_paths()
        else:
            image_paths = get_db_by_name(m).get_all_paths()
        db_size = len(image_paths)
        db_sizes[m] = db_size
        total_db_size = total_db_size + db_size
    
    # Calculate workload
    workload = total_db_size // workers
    return workload, db_sizes

# Assign work to each worker
def assign_work(cmd_single: str, db_sizes: dict, workload: int,
                model_name: str, target_name: str) -> list:
    # Determine if it is a base prediction (on self), or a cross-dataset prediction
    if target_name is None:
        target_name = model_name
        model_name = None
    
    commands = []

    db_workers = db_sizes[target_name+'w'] # TARGETw (w as workload)
    slice_start = 0
    for i in range(db_workers):
        # Current db finished, pass to the next one
        if slice_start > db_sizes[target_name] or slice_start == -1:
            break

        slice_end = slice_start + workload # note: end index is excluded in predictions

        # The last worker finish the set
        if i == db_workers -1 or slice_end > db_sizes[target_name]:
            slice_end = -1
        
        # Format the command properly
        # base prediction cmd_single only requires target_name
        if model_name is None:
            commands.append(cmd_single.format(target_name, slice_start, slice_end))
        # cross prediction cmd_single requires model_name and target_name
        else:
            commands.append(cmd_single.format(model_name, target_name, slice_start, slice_end))

        # Update starting index
        slice_start = slice_end
    return commands

def log_debug(debug: bool, workers: int, workload: int, db_sizes: dict = None):
    if debug:
        print(f'Workers  = {workers}')
        print(f'Workload = {workload}')

        if db_sizes is not None:
            print(f'Assigned work:')
            print(db_sizes)

def run_commands(commands: list, workers: int, debug: bool):
    if debug:
        print('Resulting commands:')
        for cmd in commands:
            print(cmd)

    # TODO: test
    processes = []
    if len(commands) <= workers:
        for cmd in commands:
            # Without shell=True it does not work on Windows
            # Without stdin,stout,sterr to None, it is not async
            process = subprocess.Popen(cmd.split(), shell=True, stdin=None, stdout=None, stderr=None)
            processes.append(process)
    else: # More commands than possible processes
        # Start possible processes
        for cmd in commands[0:workers]:
            process = subprocess.Popen(cmd.split(), shell=True, stdin=None, stdout=None, stderr=None)
            processes.append(process)
        # Wait till the last process has finished
        # TODO: It is naive: maybe the last process has little work to do
        processes[-1].wait()
        # Start remaining processes
        for cmd in commands[workers:]:
            process = subprocess.Popen(cmd.split(), shell=True, stdin=None, stdout=None, stderr=None)
            processes.append(process)

# Main command which groups the subcommands: single, batch
@click.group()
def cli_multipredict():
    pass

@cli_multipredict.command(name='singlem', short_help='Multiprocessing on single prediction')
@click.option('--model', '-m',
              type=click.Choice(skin_databases_names(), case_sensitive=False), required=True)
@click.option('--predict', '-p', 'predict_',
              type=click.Choice(skin_databases_names(), case_sensitive=False))
@click.option('--workers', '-w', type=int, default=-1, help = 'Number of processes, -1 for automatic')
@click.option('--debug/--no-debug', '-d', 'debug', default=False, help = 'Print more info')
def single_multi(model, predict_, workers, debug):
    pred_set = ''
    
    # Get images to predict
    if predict_ == model:
        image_paths = get_db_by_name(predict_).get_test_paths() # on same dataset, use test paths
    else:
        image_paths = get_db_by_name(predict_).get_all_paths() # on cross datasets, use all paths
    db_size = len(image_paths)

    # Calculate workload
    workers = determine_workers(workers)
    workload = db_size // workers
    log_debug(debug, workers, workload)

    cmd_single = cmd_root + '--model={} --predict={} --from={} --to={}' + pred_set
    commands = []
    # Assign work to each worker
    slice_start = 0
    for i in range(workers):
        slice_end = slice_start + workload # note: end index is excluded in predictions

        # The last worker finish the set
        if i == workers -1: # TODO: need more checks like in batch? What if slice_end > db_size?
            slice_end = -1
        
        commands.append(cmd_single.format(model, predict_, slice_start, slice_end))

        # Update starting index
        slice_start = slice_end
    
    # start processes and do not wait
    run_commands(commands, workers, debug)


@cli_multipredict.command(name='batchm', short_help='Multiprocessing on batch predictions (eg. base, cross)')
@click.option('--type', '-t', type=click.Choice(['base', 'cross', 'all']), required=True)
@click.option('--skintones/--no-skintones', 'skintones', default=False,
              help = 'Whether to predict on skintone sub-datasets')
@click.option('--workers', '-w', type=int, default=-1, help = 'Number of processes, -1 for automatic')
@click.option('--debug/--no-debug', '-d', 'debug', default=False, help = 'Print more info')
def batch_multi(type, skintones, workers, debug):
    models = skin_databases_normal
    if skintones == True:
        models = skin_databases_skintones
    
    # Models list to string list
    models = skin_databases_names(models)
    # Check if the number of workers need to be automatically determined
    workers = determine_workers(workers)

    # Determine commands to run
    commands = [] # will contain commands to be called by the terminal
    if type == 'base':
        cmd_single = cmd_root + '--model={} --from={} --to={}'

        # Calculate workload via db sizes
        workload, db_sizes = calculate_workload(models, workers, use_only_test_set=True)
        # Assign workers to each db
        db_sizes = determine_workers_per_db(db_sizes, workload, workers, debug)
        log_debug(debug, workers, workload, db_sizes)

        for m in models:
            # Assign work and concatenate the resulting commands
            commands.extend(assign_work(cmd_single, db_sizes, workload, m, None))
    elif type == 'cross':
        cmd_single = cmd_root + '--model={} --predict={} --from={} --to={}'

        # Calculate workload via db sizes
        # on cross predictions, use all paths
        workload, db_sizes = calculate_workload(models, workers, use_only_test_set=False)
        # Assign workers to each db
        db_sizes = determine_workers_per_db(db_sizes, workload, workers, debug)
        log_debug(debug, workers, workload, db_sizes)

        for m in models: # model: train dataset
            for p in models: # prediction dataset
                # In cross dataset do not predict on self
                if m == p:
                    continue
                # Assign work and concatenate the resulting commands
                commands.extend(assign_work(cmd_single, db_sizes, workload, m, p))
    else: # 'all' does either base+cross or skinbase+skincross, depending on --skintone
        #models = skin_databases 
        pass

    # start processes and do not wait
    run_commands(commands, workers, debug)
