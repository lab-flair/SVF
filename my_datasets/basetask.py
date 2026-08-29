# my_datasets/basetask.py
import random
from torch.utils.data import Dataset

class BaseTask(Dataset):
    def __init__(self, task_name, sample_mode='random', max_data_num=None, use_instruction=False, seed=0):
        super().__init__()
        self.task_name = task_name
        self.sample_mode = sample_mode
        self.max_data_num = max_data_num   
        self.use_instruction = use_instruction  
        self.task_type = None
        self.dataset = None        # raw datasets object
        self.all_data = None       # list of example dicts
        self.all_labels = None     # list of labels (ints) aligned with all_data
        self.seed = seed

    def random_sample_data(self, max_data_num, seed=0):
        """Randomly sample data (optionally class-balanced if self.all_labels is available)."""
        random.seed(self.seed if seed == 0 else seed)
        assert self.all_data is not None, "Please load data first!"

        if max_data_num < len(self.all_data):
            if (self.all_labels is None) or (self.sample_mode == 'random'):
                self.all_data = random.sample(self.all_data, max_data_num)
            else:
                # class-balanced sampling
                label_dict = {label: [] for label in set(self.all_labels)}
                for idx, label in enumerate(self.all_labels):
                    label_dict[label].append(self.all_data[idx])

                print("Number of data in each class:")
                for label, items in label_dict.items():
                    print(f"{label}: {len(items)}")

                new_all_data = []
                key_num = len(label_dict)
                per = max_data_num // key_num if key_num > 0 else max_data_num
                remaining = max_data_num

                for label, items in label_dict.items():
                    take = min(len(items), per)
                    new_all_data.extend(random.sample(items, take))
                    remaining -= take

                pool = [x for xs in label_dict.values() for x in xs]
                while remaining > 0 and len(new_all_data) < len(self.all_data):
                    cand = random.choice(pool)
                    if cand not in new_all_data:
                        new_all_data.append(cand)
                        remaining -= 1

                self.all_data = new_all_data
        else:
            print(f"Warning: max_data_num {max_data_num} >= dataset size {len(self.all_data)}; using full dataset.")

    def get_max_demonstration_token_length(self, tokenizer):
        all_data_toks = []
        for data in self.all_data:
            input_str, ans_str, _ = self.apply_template(data)
            instruct = (self.get_task_instruction() + '\n') if self.use_instruction else ""
            demonstration_str = [(instruct + input_str + ' ' + ans_str[i]) for i in range(len(ans_str))]
            all_data_toks.extend(demonstration_str)

        enc = tokenizer.batch_encode_plus(all_data_toks, return_tensors="pt", padding=True, truncation=True)
        single_max = enc['attention_mask'].sum(dim=1).max().item()

        if 'Llama' in tokenizer.name_or_path:
            cxt_max_len = 4096 if 'Llama-2' in tokenizer.name_or_path else 2048
        else:
            cxt_max_len = getattr(tokenizer, "model_max_length", 2048)
        max_demonstration_len = cxt_max_len - single_max
        print(f"Max demonstration token length: {cxt_max_len} - {single_max} = {max_demonstration_len}")
        return max_demonstration_len

    def gen_few_shot_demonstration(self, tokenizer, shot_num, max_demonstration_tok_len=1e6,
                                   add_extra_query=False, example_separator='\n',
                                   return_data_index=False, gen_example_method='normal', seed=0):
        random.seed(seed)
        assert self.all_data is not None and len(self.all_data) > 0, "Please load data first!"
        assert (shot_num == -1 or shot_num == 0 or shot_num <= len(self.all_data)), "Invalid shot_num!"

        if hasattr(self, 'class_num') and self.class_num is not None:
            assert shot_num in (-1, 0) or shot_num >= self.class_num, \
                "Shot number should be >= number of classes!"
            class_num = self.class_num
            label_dict = {label: [] for label in range(class_num)}
            for index, data in enumerate(self.all_data):
                label = self.apply_template(data)[-1]
                label_dict[label].append(index)
        else:
            class_num = None

        instruct = self.get_task_instruction() if self.use_instruction else ""
        demonstration_examples = [instruct] if len(instruct) > 0 else []
        demonstration = (instruct + example_separator) if len(instruct) > 0 else ""

        if class_num is None:
            sample_indexes = random.sample(range(len(self.all_data)), shot_num)
        else:
            per = shot_num // class_num
            sample_indexes = []
            for label in label_dict:
                sample_indexes.extend(random.sample(label_dict[label], per))
            random.shuffle(sample_indexes)

        for idx in sample_indexes:
            input_str, ans_str, label = self.apply_template(self.all_data[idx])
            ans = ans_str[label]
            new_example = input_str + ' ' + ans
            demonstration += new_example + example_separator

            if gen_example_method == 'normal':
                single = new_example
            elif gen_example_method == 'random_label':
                import random as _r
                single = input_str + ' ' + _r.choice(ans_str)
            elif gen_example_method == 'no_template':
                single = new_example
            elif gen_example_method == 'random_order':
                words = new_example.split()
                random.shuffle(words)
                single = ' '.join(words)
            else:
                raise ValueError("Unknown gen_example_method!")
            demonstration_examples.append(single + example_separator)

        if add_extra_query:
            import random as _r
            extra_query, _, _ = self.apply_template(self.all_data[_r.randint(0, len(self.all_data)-1)])
            demonstration += extra_query
            demonstration_examples.append(extra_query)

        enc = tokenizer(demonstration, return_tensors="pt", padding=True, truncation=False)
        assert enc['input_ids'].shape[1] < max_demonstration_tok_len, \
            "Few-shot demonstration too long for max_demonstration_tok_len!"
        return (demonstration, demonstration_examples, sample_indexes) if return_data_index \
               else (demonstration, demonstration_examples)

    def get_dmonstration_template(self):
        raise NotImplementedError

    def get_task_instruction(self):
        raise NotImplementedError

    def apply_template(self, data):
        raise NotImplementedError

    def print_data(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        for idx in indices:
            print(self.all_data[idx])

    def __len__(self):
        return len(self.all_data) if self.all_data is not None else 0

    def __getitem__(self, index):
        return self.all_data[index]
