import gzip
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
_CATEGORY_TO_COLORS = {
                "PARAMETER": "darkgreen",
                "OPTIMIZER_STATE": "goldenrod",
                "INPUT": "black",
                "TEMPORARY": "mediumpurple",
                "ACTIVATION": "red",
                "GRADIENT": "mediumblue",
                "AUTOGRAD_DETAIL": "royalblue",
                None: "grey",}                

_ACTION= {
    "PREEXISTING": 1,
    "CREATE" : 2,
    "INCREMENT_VERSION" : 3,
    "DESTROY" : 4,
}

_ACTION_TO_INDEX = {c: i for i, c in enumerate(_ACTION)}

_CATEGORY_TO_INDEX = {c: i for i, c in enumerate(_CATEGORY_TO_COLORS)}

def _coalesce_timeline(timeline):
        """Convert the memory timeline and categories into a memory plot
        consisting of timestamps and their respective sizes by category
        for a given device.

        Input: device
        Output: [timestamps, sizes by category]
        """
        times: list[int] = []
        sizes: list[list[int]] = []


        # def update(key, version, delta):
        #     category = (
        #         self.categories.get(key, version)
        #         if isinstance(key, TensorKey)
        #         else None
        #     )
        #     index = _CATEGORY_TO_INDEX[category] + 1
        #     sizes[-1][index] += int(delta)
        def update(category, delta):
            #index = _CATEGORY_TO_INDEX[category] + 1
            index = category + 1
            sizes[-1][index] += int(delta)
        
        t_min = -1
        for t, action,  numbytes, category in timeline:
            
            # Convert timestamps from ns to us, to match trace events.
            if t != -1:
                t = int(t / 1000)

            # Save the smallest timestamp to populate pre-existing allocs.
            if t_min == -1 or (t < t_min and t > 0):
                t_min = t

            # Handle timestep
            if len(times) == 0:
                times.append(t)
                sizes.append([0] + [0 for _ in _CATEGORY_TO_INDEX])

            elif t != times[-1]:
                times.append(t)
                sizes.append(sizes[-1].copy())
            # Handle memory and categories
            update(category, numbytes)

        times = [t_min if t < 0 else t for t in times]
        
        return times, sizes


def open_json_gz(file_path, device_str='cuda:0', title=None):

    #try:
        #with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            '''data is a list. Each item in the list is a raw memory event consisting of (timestamp, action, numbytes, category)'''
            data = json.load(f)

        mt = _coalesce_timeline(data)
        times, sizes = np.array(mt[0]), np.array(mt[1])
        # For this timeline, start at 0 to match Chrome traces.
        t_min = min(times)
        times -= t_min
        stacked = np.cumsum(sizes, axis=1) / 1024**3
        device = torch.device(device_str)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)

        # Plot memory timeline as stacked data
        fig = plt.figure(figsize=(20, 12), dpi=80)
        axes = fig.gca()
        for category, color in _CATEGORY_TO_COLORS.items():
            i = _CATEGORY_TO_INDEX[category]
            axes.fill_between(
                times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
            )
        
        fig.legend([ i for i in _CATEGORY_TO_COLORS.keys()], loc='upper right')
        # Usually training steps are in magnitude of ms.
        axes.set_xlabel("Time (ms)")
        axes.set_ylabel("Memory (GB)")
        title = "\n\n".join(
            ([title] if title else [])
            + [
                f"Max memory allocated: {max_memory_allocated / (1024**3):.2f} GiB \n"
                f"Max memory reserved: {max_memory_reserved / (1024**3):.2f} GiB"
            ]
        )
        axes.set_title(title)
        
        fig.savefig("memory.png")

        print("fig successfully created form raw memory timeline data.")
        # Now 'data' contains your JSON content as a Python object (e.g., dictionary or list)
        # You can access and process the data as needed


    # except FileNotFoundError:
    #     print(f"Error: The file '{file_path}' was not found.")
    # except json.JSONDecodeError as e:
    #     print(f"Error decoding JSON from the file: {e}")
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', default="output/train_flow_g4/2025-11-14-06-34-26/profiling/memory_timeline.raw.json.gz" )

    args = parser.parse_args()

    return args

 


if __name__ == "__main__":
    args = parse_args()
    file_path =args.file_path
    open_json_gz(file_path)