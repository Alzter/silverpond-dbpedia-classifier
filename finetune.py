from datasets import Dataset, Value, ClassLabel, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer
import numpy as np
import pandas as pd
import transformers, torch
from pandas import DataFrame
from copy import copy
import math
import warnings

transformers.set_seed(42) # Enable deterministic LLM output

def create_dataset_from_dataframe(df : DataFrame, text_column : str, labels_column : str, test_size : float | None = 0.1) -> Dataset:
    """
    Convert a DataFrame into a Dataset for pre-processing.

    Args:
        df (DataFrame): The DataFrame to convert.
        text_column (str): The column name for the input text column (X).
        labels_column (str): The column name for the output label column (y). Labels *must* be a string, not a class ID.
        test_size (float, optional): If specified, splits the dataset into train and test subsets where test_size is the ratio of the test subset. Defaults to 0.1.

    Returns:
        Dataset: The dataset.
    """
    df = df.copy() # Copy the dataframe to prevent the original being modified

    df[labels_column] = df[labels_column].map(lambda x : x.strip() if type(x) is str else x)

    # Delete any empty values
    df = df.dropna(subset=[text_column, labels_column])
    
    data = {
        "text" : df[text_column].to_list(),
        "label" : df[labels_column].to_list()
    }
    
    ds = Dataset.from_dict(data)
    ds = ds.class_encode_column("label") # Convert label from Value to ClassLabel

    if test_size is not None:
        ds = ds.train_test_split(test_size=test_size)
    
    #ds = ds.flatten_indices() # Call .flatten_indices() after .filter() otherwise .sort() takes ages.

    return ds

def select_top_n_classes(dataset : Dataset | DatasetDict, n : int = 10, labels_column : str = "label", main_subset : str = 'train', top_n_labels : list | None = None) -> Dataset:
    """
    Given a dataset, limit the total number of classes to the top ``n`` most common.
    All samples not within the top ``n`` classes are removed.
    
    NOTE: This does *not* adjust the number of samples *per* class to be equal; use ``sample_dataset()`` for that.

    Args:
        dataset (Dataset | DatasetDict): The dataset to sample.
        n (int, optional): Top ``n`` most common classes to select from the dataset. Defaults to 10.
        labels_column (str, optional): The column name for the labels in the dataset. Defaults to "label".
        main_subset (str, optional): If a ``DatasetDict`` is passed to this function (e.g., a dataset split into train/test subsets),
                                        this argument specifies *which* subset to use to get the top ``n`` classes.
                                        Defaults to 'train'.
        top_n_labels (list | None, optional): Optional arbitrary list of class labels to use. If provided, selects these class labels instead of the top ``n`` classes. Defaults to None.

    Returns:
        Dataset: The dataset with only samples from the top ``n`` classes.
    """
    if top_n_labels is None:
        # Select the top n class labels from the main dataset
        if type(dataset) is DatasetDict:
            # If the dataset is split into train/test sets,
            # use the train subset by default, since that is the largest
            main_dataset = dataset[main_subset]
        else: main_dataset = dataset
        
        labels = pd.Series(main_dataset[labels_column])
        n = max(n, 1)
        n = min(n, labels.size)
        
        # Get the top n labels from the dataset as a list
        top_n_labels = labels.value_counts().iloc[0:n].keys().to_list()
    
    # If the dataset is actually a container of datasets,
    # use recursion to sample all sub-datasets
    if type(dataset) is DatasetDict:
        dataset = copy(dataset) # Shallow copy the DatasetDict to prevent the original being modified
        for subset in dataset.keys():
            dataset[subset] = select_top_n_classes(dataset[subset], labels_column=labels_column, n=n, top_n_labels = top_n_labels)
        #dataset = dataset.flatten_indices() # Call .flatten_indices() after .filter() otherwise .sort() takes ages.
        return dataset
    
    dataset = dataset.filter(lambda x : x[labels_column] in top_n_labels)

    # If the labels column is a ClassLabel, we also have to update the label names to match the new classes.
    if type(dataset.features[labels_column]) is ClassLabel:

        # Convert class label column into raw strings.
        label_names = dataset.features[labels_column].names
        dataset = dataset.cast_column(labels_column, Value(dtype='string'))
        dataset = dataset.map( lambda sample : _label_to_string(sample, label_names, label_column=labels_column) )
        
        # Cast class label column back into a ClassLabel.
        dataset = dataset.class_encode_column(labels_column)

    #dataset = dataset.flatten_indices() # Call .flatten_indices() after .filter() otherwise .sort() takes ages.
    return dataset

def _get_n_samples_per_class(dataset : Dataset, n : int, labels_column : str, shuffle:bool=True, seed:int=0) -> Dataset:
    """
    Given a dataset, obtain a smaller dataset containing the first **n** samples from each class.

    Args:
        dataset (Dataset): The dataset to sample.
        labels_column (str): The column name for the labels in the dataset.
        n (int): How many samples from each class to extract.
        shuffle (bool): Whether to sort the final result by class or randomly.
        seed (int, optional): RNG seed. Defaults to 0.

    Returns:
        Dataset: The sample of the dataset.
    """

    if labels_column not in dataset.features.keys(): raise ValueError(f"Dataset has no column: {labels_column}")


    ds_sorted = dataset.flatten_indices().sort(labels_column) # BUG: .sort() takes forever if called after .filter() unless you call .flatten_indices()

    _, class_indices = np.unique(ds_sorted[labels_column], return_index=True)

    # Get the number of samples from the least common class.
    num_samples_in_minority_class = pd.Series(ds_sorted[labels_column]).value_counts().min()

    # Ensure n is not greater than the number of samples per class.
    samples_per_class = np.diff(class_indices).min()
    n = min(n, samples_per_class)
    n = max(n, 1)

    # Undersample if needed to ensure an equal number of samples per class.
    if n > num_samples_in_minority_class:
        warnings.warn(f"\nCannot sample {n} samples per class equally because some classes have fewer samples.\nSampling {num_samples_in_minority_class} samples per class instead.\n")
    n = min(n, num_samples_in_minority_class)

    class_indices = np.array([list(range(index, index + n)) for index in class_indices])
    class_indices = class_indices.flatten()

    sample = ds_sorted.select(class_indices)

    if shuffle: sample = sample.shuffle(seed=seed)
    
    return sample

def sample_dataset(dataset : Dataset | DatasetDict, labels_column : str = "label", ratio : float = None, size : int = None, samples_per_class : int = None, shuffle : bool = True, seed:int=0) -> Dataset:
    """
    Given a dataset, return a smaller dataset with an equal number of samples per class.
    
    There are 3 methods of sampling a dataset:
    - **ratio**: Specify the size of the sample as a percentage of the original dataset from 1 - 0.
    - **size**: Specify how many items the sample should have in total.
    - **samples_per_class**: Specify how many items each class should have inside the sample.

    Args:
        dataset (Dataset | DatasetDict): The dataset to sample.
        labels_column (str, optional): The column name for the labels in the dataset. Defaults to "label".
        ratio (float, optional): What percentage of the dataset to sample from 1-0. Defaults to None.
        size (int, optional): Number of items the new dataset should have. Defaults to None.
        samples_per_class (int, optional): Number of items per class the new dataset should have. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the dataset before and after sampling it. Defaults to True.
        seed (int, optional): RNG seed. Defaults to 0.

    Returns:
        Dataset: The sample of the dataset.
    """

    # If the dataset is actually a container of datasets,
    # use recursion to preprocess all sub-datasets
    if type(dataset) is DatasetDict:
        dataset = copy(dataset) # Shallow copy the DatasetDict to prevent the original being modified
        for subset in dataset.keys():
            dataset[subset] = sample_dataset(dataset[subset], labels_column, ratio, size, samples_per_class, shuffle, seed)
        return dataset

    if shuffle: dataset=dataset.shuffle(seed=seed)
    
    if labels_column not in dataset.features.keys(): raise ValueError(f"Dataset has no column: {labels_column}")

    if ratio is None and size is None and samples_per_class is None:
        raise ValueError("Either ratio, size, or samples_per_class must be given.")

    if samples_per_class is None:
        if size is not None:
            ratio = size / dataset.num_rows
        ratio = max(ratio, 0)
        ratio = min(ratio, 1)

        num_labels = np.unique(dataset[labels_column]).size
        
        samples_per_class = dataset.num_rows / num_labels * ratio
        samples_per_class = int(math.floor(samples_per_class))

    return _get_n_samples_per_class(dataset, samples_per_class, labels_column, shuffle=shuffle, seed=seed)

def _format_dataset(examples : Dataset) -> dict:
    """
    Converts a Dataset in [prompt/completion format](https://huggingface.co/docs/trl/main/en/dataset_formats#standard)
    to [conversational format](https://huggingface.co/docs/trl/main/en/dataset_formats#conversational).
    This is necessary because the trl library no longer supports prompt/completion datasets for LLM fine-tuning.
    Adapted from [Daniel Voigt Godoy](https://github.com/dvgodoy/FineTuningLLMs/blob/main/Chapter0.ipynb).
    
    Args:
        examples (Dataset): Dataset in prompt/completion format.

    Returns:
        Dataset: Dataset in conversational format.
    """
    if isinstance(examples["prompt"], list):
        output_texts = []
        for i in range(len(examples["prompt"])):
            converted_sample = [
                {"role": "user", "content": examples["prompt"][i]},
                {"role": "assistant", "content": examples["completion"][i]},
            ]
            output_texts.append(converted_sample)
        return {'messages': output_texts}
    else:
        converted_sample = [
            {"role": "user", "content": examples["prompt"]},
            {"role": "assistant", "content": examples["completion"]},
        ]
        return {'messages': converted_sample}

def _label_to_string(sample : dict, class_label_names : list, label_column : str = "completion") -> dict:
    """
    Given a sample from a supervised text classification dataset
    in prompt/completion format, replace the class label ID with
    a label name string. The class label column must already have
    a string data type.

    Args:
        sample (dict): The data sample in prompt/completion format with string data type.
        class_label_names (list): A list of all class label names.

    Returns:
        dict: The data sample with class ID replaced with label name.
    """
    sample[label_column] = class_label_names[ int(sample[label_column]) ]
    return sample

def preprocess_dataset(dataset : Dataset | DatasetDict, text_column : str = "text", labels_column : str = "label") -> tuple[Dataset, list]:
    """
    Pre-process a supervised text-classification dataset into a format usable for fine-tuning.

    Args:
        dataset (Dataset | DatasetDict): A supervised text-classification dataset.
        text_column (str): The column name for the input text column (X).
        labels_column (str, optional): The column name for the output label column (y). Defaults to "label".
    
    Returns:
        formatted_dataset (Dataset): The dataset in conversational format.
        label_names (list): The list of class label names.
    """
    
    # If the dataset is actually a container of datasets,
    # use recursion to preprocess all sub-datasets
    if type(dataset) is DatasetDict:
        label_names = []
        dataset = copy(dataset) # Shallow copy the DatasetDict to prevent the original being modified
        for subset in dataset.keys():
            dataset[subset], new_labels = preprocess_dataset(dataset[subset], text_column, labels_column)
            if len(new_labels) > len(label_names): label_names = new_labels
        return dataset, label_names

    for column in [text_column, labels_column]:
        if column not in dataset.features.keys(): raise ValueError(f"Dataset has no column: {column}")

    # Select only the text and label columns.
    dataset = dataset.select_columns([text_column,labels_column])

    # Convert the dataset into prompt/completion format.
    dataset = dataset.rename_column(text_column, "prompt")
    dataset = dataset.rename_column(labels_column, "completion")

    # Map the class label column from integer to string.
    if type(dataset.features['completion']) is ClassLabel:
        label_names = [n.strip() for n in dataset.features['completion'].names]
        # Cast label column from int to str.
        dataset = dataset.cast_column("completion", Value(dtype='string'))
        # Replace all class label IDs with label names.
        dataset = dataset.map( lambda sample : _label_to_string(sample, label_names) )
    else:
        label_names = [n.strip() for n in list(np.unique(dataset['completion']))]

    # Convert the dataset into conversational format
    dataset = dataset.map(_format_dataset).remove_columns(['prompt', 'completion'])

    return dataset, label_names

def finetune(model : AutoModelForCausalLM, tokenizer : AutoTokenizer, train_dataset : Dataset, lora_config : LoraConfig, sft_config : SFTConfig, output_dir : str | None = None) -> None:
    """Fine-tune an LLM using LoRA and save the resulting adapters in ``output_dir``. The LLM specified in ``model`` **will** be modified by this function.

    Args:
        model (AutoModelForCausalLM): The LLM to fine-tune, which will be modified by this function. Use ``AutoModelForCausalLM.from_pretrained(model_name)`` to instantiate.
        tokenizer (AutoTokenizer): The tokenizer to use. Should come with the LLM. Use ``AutoTokenizer.from_pretrained(model_name)`` to instantiate.
        train_dataset (Dataset): The dataset of training samples to fine-tune the model on. You must pre-process this dataset using ``preprocess_dataset`` before calling this method.
        lora_config (LoraConfig): LoRA hyperparameters, including the rank of the adapters and the scaling factor.
        sft_config (SFTConfig): Fine-tuning training configuration, including number of epochs, checkpoints, etc.
        output_dir (str, optional): Where to save the fine-tuned model to. Defaults to ``sft_config.output_dir``.
    """
    if output_dir is None:
        output_dir = sft_config.output_dir
    
    if type(model) is not AutoPeftModelForCausalLM:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)

def load_finetuned_llm(model_directory : str, device_map : str = "cuda:0", quantized:bool = True) -> tuple[AutoPeftModelForCausalLM, AutoTokenizer]:
    """
    Load a finetuned LLM from disk.

    Args:
        model_directory (str): Where to load the fine-tuned model.
        device_map (str, optional): Which device to load the fine-tuned model onto. Defaults to "cuda:0".
        quantized (bool, optional): Whether to load the model with 4-bit quantization. Defaults to True.

    Returns:
        model (AutoPeftModelForCausalLM): The fine-tuned LLM.
        tokenizer (AutoTokenizer): The tokenizer (unchanged from the base model).
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16
    ) if quantized else None

    config = PeftConfig.from_pretrained(model_directory)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = AutoPeftModelForCausalLM.from_pretrained(model_directory, device_map=device_map, quantization_config=bnb_config)

    return (model, tokenizer)

def _format_prompt(prompt : str | dict, tokenizer : AutoTokenizer) -> str:
    """
    Convert an LLM prompt into string format with a chat template
    and special tokens.

    Args:
        prompt (str | dict): The prompt for the LLM.
                            You can use a string for a simple user prompt or a [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
                            if you want to include a system prompt and/or prior chat history.
        tokenizer (AutoTokenizer): The tokenizer to use. Should come with the LLM. Use ``AutoTokenizer.from_pretrained(model_name)`` to instantiate.

    Returns:
        prompt (str): The prompt with chat template applied converted to string format using special tokens.
    """
    if type(prompt) is str:
        prompt = [{"role": "user", "content": prompt}]
    
    prompt = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    return prompt

def generate(
    prompt : str | dict,
    model : AutoModelForCausalLM,
    tokenizer : AutoTokenizer,
    max_new_tokens : int = 64,
    response_only : bool = True,
    skip_special_tokens : bool = True,
    do_sample : bool = False,
    temperature : float | None = None,
    top_p : float | None = None,
    top_k : float | None = None,
    kwargs : dict = {}
    ) -> str:
    """
    Generate an LLM response to a given query.

    Args:
        prompt (str | dict): The prompt for the LLM.
                            You can use a string for a simple user prompt or a [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
                            if you want to include a system prompt and/or prior chat history.
        model (AutoModelForCausalLM): The LLM to use. Use ``AutoModelForCausalLM.from_pretrained(model_name)`` to instantiate.
        tokenizer (AutoTokenizer): The tokenizer to use. Should come with the LLM. Use ``AutoTokenizer.from_pretrained(model_name)`` to instantiate.
        max_new_tokens (int, optional): Maximum number of tokens for the model to output. Defaults to 64.
        response_only (bool, optional): If True, excludes all previous messages from the output. Defaults to True.
        skip_special_tokens (bool, optional): If True, removes model special tokens from the output. Defaults to True.
        do_sample (bool, optional): If False, enables deterministic generation. Defaults to False.
        temperature (float, optional): Higher = greater likelihood of low probability words. Leave empty if ``do_sample`` is False. Defaults to None.
        top_p (float, optional): If set to < 1, only the smallest set of most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation. Leave empty if ``do_sample`` is False. Defaults to None.
        top_k (float, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering. Leave empty if ``do_sample`` is False. Defaults to None.
        kwargs (dict, optional): Additional parameters to pass into ``model.generate()``. Defaults to {}.
    Returns:
        response (str): The LLM's response.
    """

    # Convert user query into a formatted prompt
    prompt = _format_prompt(prompt, tokenizer=tokenizer)

    # Tokenize the formatted prompt
    tokenized_input = tokenizer(prompt,
                                add_special_tokens=False,
                                return_tensors="pt").to(model.device)
    model.eval()

    # Generate the response
    generation_output = model.generate(**tokenized_input,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=do_sample,
                                       temperature=temperature,
                                       top_p = top_p,
                                       top_k = top_k,
                                       **kwargs)

    # If required, remove the tokens belonging to the prompt
    if response_only:
        input_length = tokenized_input['input_ids'].shape[1]
        generation_output = generation_output[:, input_length:]
    
    # Decode the tokens back into text
    output = tokenizer.batch_decode(generation_output, skip_special_tokens=skip_special_tokens)[0]
    return output



    

