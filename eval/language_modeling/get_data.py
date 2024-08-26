from datasets import load_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path_or_name", type=str, nargs='+',required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--save_format", type=str, default="json", choices=["json", "llama-bin"])
    return parser.parse_args()

def main():
    
    args = parse_args()
    dataset_path_or_name_list = args.dataset_path_or_name
    split = args.split
    for dataset_path_or_name in dataset_path_or_name_list:
        dataset_name = dataset_path_or_name.split('/')[-1]
        if dataset_name == "wikitext":
            dataset = load_dataset(dataset_path_or_name,"wikitext-103-raw-v1",split=split,num_proc=args.num_proc,cache_dir=".")
        else:
            dataset = load_dataset(dataset_path_or_name, split)
        #save_to_disk(dataset, f"{args.output_dir}/{data}_{split}")


if __name__ == "__main__":
    main()