import os
import subprocess
import glob
import pandas as pd
import glob
import io
import json

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

manifest_path = '/mnt/hdd0/brain_observatory_cache/manifest.json'

boc = BrainObservatoryCache(manifest_file=manifest_path)

containers = boc.get_experiment_containers()

container_ids = [c['id'] for c in containers]
container_ids.sort()
container_id = container_ids[2]

exps = boc.get_ophys_experiments(experiment_container_ids=[container_id])
experiment_id = exps[0]['id']

temp_directory = '/mnt/hdd0/2p_conversion/' + str(experiment_id)

output_path = temp_directory + '/' + str(experiment_id) +'.nwb'

json_directory = '/mnt/md0/data/json_files'

module = 'allensdk.brain_observatory.convert_to_nwb2'

input_json = os.path.join(json_directory, str(experiment_id) + '-' + module + '-input.json')
output_json = os.path.join(json_directory, str(experiment_id) + '-' + module + '-output.json')

dictionary = {
    'experiment_id' : experiment_id,
    'manifest_path' : manifest_path,
    'temp_directory' : temp_directory,
    'output_path' : output_path
}

with io.open(input_json, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dictionary, ensure_ascii=False, sort_keys=True, indent=4))

print('Running ' + module)

command_string = ["python", "-W", "ignore", "-m", module, 
                "--input_json", input_json,
                "--output_json", output_json]

subprocess.check_call(command_string)



