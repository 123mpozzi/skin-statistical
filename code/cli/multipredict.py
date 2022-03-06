import subprocess
import time
import psutil
import click
from utils.db_utils import *
from utils.logmanager import *

# Parallelize the predictions by calling predict.py multiple times


cmd_root = 'python main.py single '

def determine_workers(workers: int) -> int:
    '''
    If workers == -1, number of workers is equal to the
    number of system's physical cores
    '''
    if workers == -1:
        workers = psutil.cpu_count(logical=False)
    return workers

def determine_tasks_per_db(db_sizes: dict, workload: int,
                                total_db_size: int, debug: bool) -> dict:
    '''
    Try to assign a proper number of tasks per each dataset

    eg: Workload = 300.
        : If db1 has 200  images  -> 1 task
        : If db2 has 1000 images  -> 4 tasks
    
    Return a dict containing each db with their size and tasks assigned
    '''
    remaining_tasks = total_db_size // workload
    
    while remaining_tasks > 0:
        # copy because it is being edited in the loop
        for entry in db_sizes.copy():
            if debug:
                info(db_sizes)
                info(remaining_tasks)
            # Just consider db entries, not tasks-related ones
            if entry[-1] == 'w': # (if string ending with 'w')
                continue

            key = entry+'w' # key containing db + number of tasks (eg. ECUw : 1)

            if remaining_tasks != 0:
                # db need more work than a workload unit
                if db_sizes[entry] > workload:
                    # Add a task
                    if key in db_sizes:
                        db_sizes[key] = int(db_sizes[key]) + 1
                    else:
                        db_sizes[key] = 1
                    remaining_tasks = remaining_tasks -1
                # A workload unit is enough
                else:
                    # If all work needed in current db is already assigned, continue
                    if key in db_sizes and db_sizes[key] == 1:
                        continue

                    # Add a task
                    db_sizes[key] = 1
                    remaining_tasks = remaining_tasks -1
            else:
                break
    return db_sizes

def calculate_workload(models: list, workers: int, use_only_test_set: bool = False) -> list:
    '''
    Calculate the workload by measuring the size of each db

    Aim for small tasks occupying less than 10 minutes (depending on images size)
    or else big dataset could occupy a worker for a very long time,
    while other workers are sleeping.

    Also aim for a number of tasks that is multiple of the number of workers
    so that they can always run in parallel

    Return also a dict containing each db with their size
    '''
    assert workers > 0, 'Number of workers is negative! Is it using the default of -1? Call determine_workers(workers)'

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
    
    workload = total_db_size // workers
    j = 2 # each worker does 2 tasks
    while workload > 300: # Aim for small tasks occupying less than 10 minutes
        workload = total_db_size // (workers *j)
        j = j+1 # each worker does 3 tasks, etc..

    return workload, db_sizes, total_db_size

def generate_commands(cmd_single: str, target_size: int, target_tasks: int, workload: int,
                model_name: str, target_name: str, bar_position: int) -> list:
    '''
    Translate tasks regarding a given target dataset into `singlepredict.py` commands

    Arguments
    ---
    cmd_single: `singlepredict.py` command to format with other arguments
    target_size: size of target dataset in images
    target_tasks: number of tasks assigned to target dataset
    workload: the workload
    model_name: the model name
    target_name: the target name
    bar_position: where the `tqdm` progress bar appears on console
    '''

    # Determine if it is a base prediction (on self), or a cross-dataset prediction
    if target_name is None:
        target_name = model_name
        model_name = None
    
    commands = []

    slice_start = 0
    for i in range(target_tasks):
        # Current db finished, pass to the next one
        if slice_start > target_size or slice_start == -1:
            break

        slice_end = slice_start + workload # note: end index is excluded in predictions

        # The last worker finish the set
        if i == target_tasks -1 or slice_end > target_size:
            slice_end = -1
        
        # Format the command properly
        # base prediction cmd_single only requires target_name
        if model_name is None:
            commands.append(cmd_single.format(target_name, slice_start, slice_end, bar_position))
        # cross prediction cmd_single requires model_name and target_name
        else:
            commands.append(cmd_single.format(model_name, target_name, slice_start, slice_end, bar_position))
        
        # Update bar position
        bar_position = bar_position + 1
        # Update starting index
        slice_start = slice_end
    return commands, bar_position

def log_debug(debug: bool, workers: int, workload: int, db_sizes: dict = None):
    if debug:
        info(f'Workers  = {workers}')
        info(f'Workload = {workload}')

        if db_sizes is not None:
            info(f'Assigned work:')
            info(db_sizes)

# Credit to https://stackoverflow.com/a/50560686
def clear_console():
    print("\033[H\033[J", end="")

def run_commands(commands: list, workers: int, debug: bool):
    '''
    Run the given commands list

    The program will automatically handle the executions even
    if there are more commands than workers

    It also tries to print the status of each alive worker via `tqdm`
    '''
    if debug:
        info(f'Resulting commands: {len(commands)}')
        for cmd in commands:
            info(cmd)

    procs_list = []
    alive_procs = 1
    # While there are still tasks running
    while alive_procs > 0:
        clear_console() # clear console or else progress bars are buggy

        # Print tasks status
        print(f'TASKS REMAINING: {len(commands)}')
        print(f'\n\nWORKERS STATUS, ALIVE={alive_procs}')
        
        # If there is a free slot among workers
        if len(commands) > 0 and alive_procs < workers:
            # Pop one command from the list
            cmd = commands.pop()

            # Run a command
            # without shell=True it does not work on Windows
            # without stdin,stout,sterr to None, it is not async
            process = subprocess.Popen(cmd.split(), shell=True, stdin=None, stdout=None, stderr=None)
            procs_list.append(psutil.Process(process.pid))
        # If the command list is empty it means the program is
        # waiting for tasks to finish (cannot pop on empty list)
        else:
            time.sleep(1.5) # be lighter than a while true
        
        # Update tasks status
        gone, alive = psutil.wait_procs(procs_list, timeout=3)
        alive_procs = len(alive)

def gen_base_cmds(models: list, workers: int, debug: bool = False):
    '''Return a list containing single commands needed to perform base dataset predictions'''
    commands = []
    cmd_single = cmd_root + '--model={} --from={} --to={} --bar={}'

    # Calculate workload and assign tasks
    workload, db_sizes, total_db_size = calculate_workload(models, workers, use_only_test_set=True)
    db_sizes = determine_tasks_per_db(db_sizes, workload, total_db_size, debug)
    log_debug(debug, workers, workload, db_sizes)

    bar_position = 0
    for m in models:
        # Assign work and concatenate the resulting commands
        cmds_on_target, bar_position = generate_commands(cmd_single, db_sizes[m], db_sizes[m+'w'], workload, m, None, bar_position)
        commands.extend(cmds_on_target)
    return commands

def gen_cross_cmds(models: list, workers: int, debug: bool = False):
    '''Return a list containing single commands needed to perform cross dataset predictions'''
    commands = []
    cmd_single = cmd_root + '--model={} --predict={} --from={} --to={} --bar={}'

    # Calculate workload and assign tasks

    # on cross predictions, use all paths
    workload, db_sizes, total_db_size = calculate_workload(models, workers, use_only_test_set=False)
    db_sizes = determine_tasks_per_db(db_sizes, workload, total_db_size, debug)
    log_debug(debug, workers, workload, db_sizes)

    bar_position = 0
    for m in models: # model: train dataset
        for p in models: # prediction: target dataset
            # In cross dataset do not predict on self
            if m == p:
                continue
            # Assign work and concatenate the resulting commands
            cmds_on_target, bar_position = generate_commands(cmd_single, db_sizes[m], db_sizes[m+'w'], workload, m, p, bar_position)
            commands.extend(cmds_on_target)
    return commands

# Main command which groups the subcommands: single, batch
@click.group()
def cli_multipredict():
    pass

@cli_multipredict.command(name='singlem', short_help='Multiprocessing on single prediction')
@click.option('--model', '-m',
              type=click.Choice(skin_databases_names(get_models()), case_sensitive=False), required=True)
@click.option('--predict', '-p', 'predict_',
              type=click.Choice(skin_databases_names(get_datasets()), case_sensitive=False))
@click.option('--workers', '-w', type=int, default=-1, help = 'Number of processes, default is auto')
@click.option('--debug/--no-debug', '-d', 'debug', default=False, help = 'Print more info')
def single_multi(model, predict_, workers, debug):
    # prediction on self
    if predict_ is None:
        predict_ = model

    models = [predict_]
    # Check if the number of workers need to be automatically determined
    workers = determine_workers(workers)

    # Get images to predict
    if predict_ == model:
        # on same dataset, use test paths
        workload, db_sizes, total_db_size = calculate_workload(models, workers, use_only_test_set=True)
    else:
        # on cross datasets, use all paths
        workload, db_sizes, total_db_size = calculate_workload(models, workers, use_only_test_set=False)

    # Calculate workload and assign tasks
    db_sizes = determine_tasks_per_db(db_sizes, workload, total_db_size, debug)
    task_number = db_sizes[predict_ + 'w']
    log_debug(debug, workers, workload, db_sizes)

    commands = []
    cmd_single = cmd_root + '--model={} --predict={} --from={} --to={} --bar={}'

    # Translate tasks to commands
    slice_start = 0
    for i in range(task_number):
        slice_end = slice_start + workload # note: end index is excluded in predictions

        # The last worker finish the set
        if i == task_number -1 or slice_end > total_db_size:
            slice_end = -1
        
        commands.append(cmd_single.format(model, predict_, slice_start, slice_end, i))

        # Update starting index
        slice_start = slice_end
    
    # start processes and do not wait
    run_commands(commands, workers, debug)

@cli_multipredict.command(name='batchm', short_help='Multiprocessing on batch predictions (eg. base, cross)')
@click.option('--type', '-t', type=click.Choice(['base', 'cross', 'all']), required=True)
@click.option('--dataset' , '-d',  multiple=True,
              type=click.Choice(skin_databases_names(get_models()), case_sensitive=False), required = True,
              help = 'Datasets to use (eg. -d ECU -d HGR_small -d medium)')
@click.option('--workers', '-w', type=int, default=-1, help = 'Number of processes, default is auto')
@click.option('--debug/--no-debug', '-d', 'debug', default=False, help = 'Print more info')
def batch_multi(type, dataset, workers, debug):
    models = dataset
    
    # Models list to string list
    models = skin_databases_names(models)
    # Check if the number of workers need to be automatically determined
    workers = determine_workers(workers)

    assert models in get_datasets(), 'Not all selected models do have a dataset folder!'

    # Determine commands to run
    if type == 'base':
        commands = gen_base_cmds(models, workers, debug=debug)
    elif type == 'cross':
        commands = gen_cross_cmds(models, workers, debug=debug)
    else: # 'all' does either base+cross or skinbase+skincross, depending on --skintone
        commands = gen_base_cmds(models, workers, debug=debug)
        commands.extend(gen_cross_cmds(models, workers, debug=debug))
        pass

    # start processes and do not wait
    run_commands(commands, workers, debug)
