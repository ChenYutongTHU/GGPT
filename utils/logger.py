import os, json
import torch.distributed as dist
import torch
class EvalLogger():
    def __init__(self,dirname):
        os.makedirs(dirname, exist_ok=True)
        self.dirname = dirname
        self.dic = {}
        self.dataset2average = None
    def write(self, metric_dict, prefix, dataset_key, seq_key):
        if dataset_key not in self.dic:
            self.dic[dataset_key] = {}
        if seq_key not in self.dic[dataset_key]:
            self.dic[dataset_key][seq_key] = {}
        for m,v in metric_dict.items():
            if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel()==1):
                self.dic[dataset_key][seq_key][f'{prefix}/{m}'] = float("{:.3g}".format(v))
    def save(self, ddp_sync=True):
        os.makedirs(os.path.join(self.dirname,'per_rank'), exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        with open(os.path.join(self.dirname,'per_rank', f'rank{rank}.json'),'w') as f:
            json.dump(self.dic, f, indent=4)
        
        if ddp_sync:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                # Read and merge all other ranks' json files
                dic_gathered = {}
                for r in range(world_size):
                    rank_filename = os.path.join(self.dirname,'per_rank', f'rank{r}.json')
                    with open(rank_filename,'r') as f:
                        dic_r = json.load(f)
                    for dataset_key in dic_r:
                        if dataset_key not in dic_gathered:
                            dic_gathered[dataset_key] = {}
                        for seq_key in dic_r[dataset_key]:
                            if seq_key in dic_gathered[dataset_key]:
                                print(f"Warning: duplicate seq_key {seq_key} in dataset {dataset_key} from rank {r}")
                            dic_gathered[dataset_key][seq_key] = dic_r[dataset_key][seq_key]
                with open(os.path.join(self.dirname, f'all_ranks.json'),'w') as f:
                    json.dump(dic_gathered, f, indent=4)
                # Compute per dataset average
                dataset2average = {}
                for dataset_key in dic_gathered:
                    average = {}
                    count = {}
                    for seq_key in dic_gathered[dataset_key]:
                        for m,v in dic_gathered[dataset_key][seq_key].items():
                            if m not in average:
                                average[m] = 0.0
                                count[m] = 0
                            average[m] += v
                            count[m] += 1
                    for m in average:
                        average[m] = float("{:.3g}".format(average[m]/count[m]))
                    dataset2average[dataset_key] = average
                    #sorted average by key
                    dataset2average[dataset_key] = dict(sorted(dataset2average[dataset_key].items()))
                with open(os.path.join(self.dirname, f'average.json'),'w') as f:
                    json.dump(dataset2average, f, indent=4)
                self.dataset2average = dataset2average
                return dataset2average
            else:
                self.dataset2average = None
                return None
        else:
            return None
