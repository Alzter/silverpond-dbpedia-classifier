from dataclasses import dataclass
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, shutil, re
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt
import finetune as ft
import pandas as pd
import numpy as np

@dataclass
class EvaluationConfig:
    """
    Defines how LLMs should be instructed to classify each text sample in a classification dataset.
    You can specify different configurations for different prompting techniques.

    NOTE: For each sample in the dataset, the LLM must output the *name* of the predicted class in its response.

    Args:
        name (str): The name of your classification technique, e.g., "Chain-of-Thought 2-shot" or "Zero-shot" or "Fine-tuned".
        max_tokens (int): How many tokens the LLM is allowed to produce to classify each sample.
                          If you are planning on having your LLM output *just* the class label,
                          you can set this value to 1. The LLM will only return the first few
                          letters of the class label, but this is usually enough to identify
                          which label it selected. See ``_get_class_id_from_model_response()`` for implementation details.
        prompt (str, optional): Optional system prompt to give the LLM before each text sample. Use to provide the LLM with classification instructions. Leave empty for fine-tuned models.
    """
    name : str
    max_tokens : int
    prompt : str | None = None
    # extractor_method : func
        
@dataclass
class EvaluationResult:
    """
    Raw LLM text classification evaluation results produced from ``evaluate.evaluate()``.

    Args:
        config (EvaluationConfig): 
        texts (list[str]): 
        labels_pred (list[int]): 
        labels_true (list[int]): 
        label_names (list[str]): 
        llm_responses (list[str]): 
    """
    config : EvaluationConfig
    texts : list[str]
    labels_pred : list[int]
    labels_true : list[int]
    label_names : list[str]
    llm_responses : list[str]

    def get_answers(self, incorrect_only : bool = False) -> pd.DataFrame:
        """
        Given raw LLM text classification evaluation data, return a DataFrame of the LLM's answers to each sample in human-readable format.

        Args:
            incorrect_only (bool, optional): Whether to only include the LLM's incorrect answers. Defaults to False.

        Returns:
            pd.DataFrame: A table containing each sample in the evaluation dataset, the LLM's response to each sample, and the predicted/actual labels.
        """
        # Cast labels from int (class ID) -> str (class name)
        y_pred = [self.label_names[id] for id in self.labels_pred]
        y_true = [self.label_names[id] for id in self.labels_true]
        
        answers = {
        "Text" : np.array(self.texts),
        "Predicted Label" : np.array(y_pred),
        "True Label" : np.array(y_true),
        "LLM Response" : np.array(self.llm_responses)
        }
        
        answers = pd.DataFrame(answers)

        if incorrect_only: answers = answers.loc[answers['Predicted Label'] != answers['True Label']]

        return answers
        
    def save(self, output_dir : str | None = None) -> None:
        """
        Creates human-readable results from raw LLM evaluation data.

        The following files are produced by this method:

        1. Confusion matrix (``confusion_matrix.png``):
        Graph visualisation of the LLM's accuracy.
        
        2. Classification report (``evaluation.csv``):
        Report of the LLM's accuracy, precision, recall, and F1 score for all classes.
        
        3. LLM answer data (``answers.csv, answers_incorrect.csv``):
        A table containing all LLM responses and a table containing only the incorrect responses.

        Args:
            output_dir (str, optional): Which folder to save the results into. If not given, defaults to ``output/<name_of_technique>`` where ``name_of_technique`` is the ``name`` parameter of the ``EvaluationConfig``.
        """
        
        if output_dir is None:
            # Make result name file safe
            result_path_name = self.result.name.lower().strip().replace(" ", "_")
            # Remove all non-alphanumeric characters
            result_path_name = "".join(c for c in filename if c.isalnum() or c in ["-", "_", " "])

            output_dir = os.path.join( "output", result_path_name )

        # shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        y_pred, y_true, label_names = self.labels_pred, self.labels_true, self.label_names

        # Calculate accuracy, precision, recall, and F1 score
        classif_report = classification_report(y_true, y_pred, zero_division=0.0, output_dict=True)
        classif_report = pd.DataFrame(classif_report).transpose()

        # Determine how many class labels will be shown in the output
        # by finding how many were used in the result
        threshold = np.max([y_pred, y_true]) + 1
        label_names = label_names[0:threshold]

        cm = confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='true')

        disp = ConfusionMatrixDisplay(cm, display_labels=label_names).plot(
            cmap = plt.cm.Blues,
            xticks_rotation='vertical',
            text_kw={'fontsize': 6},
            values_format='.0%'
        )

        disp.ax_.set_title( result.config.name )

        answers = self.get_answers(incorrect_only=False)
        incorrect_answers = self.get_answers(incorrect_only=True)

        classif_report.to_csv( os.path.join(output_dir, "evaluation.csv") )
        answers.to_csv( os.path.join(output_dir, "answers.csv"), escapechar="\\" )
        incorrect_answers.to_csv( os.path.join(output_dir, "incorrect_answers.csv"), escapechar="\\" )
        plt.savefig( os.path.join(output_dir, "confusion_matrix.png"), dpi=200, bbox_inches='tight' )

        plt.show()

def _get_class_id_from_model_response(model_response : str, label_names : list) -> int:
    """
    After getting an LLM to perform text classification,
    this function is used to extract class IDs from raw
    LLM outputs.

    Given a raw LLM output ``model_response``, attempt to find the name of
    a matching class label from ``label_names`` from the
    output and return its ID (position in the list).

    Assume that ``label_names`` contains a final
    "I don't know" entry for all examples that the LLM
    could not generate a class label for.

    The method to extract class IDs has 3 steps:
    1. Try to directly match the output with an item in ``label_names``.
    2. If ``model_response`` only contains the first few letters of a class label, try to match it with a truncated class label.
       This allows you to run LLM evaluation with fewer max tokens than needed for the LLM to fully write each class label,
       which greatly reduces the time it takes for the model to classify each sample while retaining accuracy.
    3. Try to find the last matching class label name from the model's response using RegEx.
       This is useful for CoT prompting, where the model may end the answer with the class label it decided on.

    Args:
        model_response (str): The raw LLM output, ideally containing a class label.
        label_names (list): List of class label names with an additional final entry for unknown cases.

    Returns:
        class_id (int): The predicted class ID, or the final class ID if the LLM did not answer with any class label.
    """

    # Return a direct match if possible
    try:
        class_id = label_names.index(model_response)
        return class_id
    except Exception as e:
        pass

    model_response = model_response.lower().strip()
    
    # Match class labels if model_response is truncated.
    # E.g., "mean" -> "meanoftransportation" -> 5
    # This allows us to match labels even we don't
    # have enough tokens to write the entire name.
    # This allows us to optimise the evaluation
    # procedure by using lower max tokens.
    for i, label in enumerate(label_names):
        
        label_truncated = label.lower().replace(" ", "")[0:len(model_response)]
        if model_response == label_truncated:
            return i
    
    # Concatenate all label names using boolean OR.
    match = "|".join(label_names).lower().replace(" ", r"\s*")

    # Find all instances of label name strings within the base string.
    matches = re.findall(match, model_response)

    # If the string contains at least one instance of a class label:
    if len(matches) > 0:
        # Get the last matching label from the string.
        final_match = matches[-1]

        # Remove all capitalisation, non-alphabetic characters, and whitespace
        labels_sanitised = [re.sub("[^a-z]", "", label.lower()) for label in label_names]
        match_sanitised = re.sub("[^a-z]", "", final_match.lower())

        # Return the matching class ID for the label.
        class_id = labels_sanitised.index(match_sanitised)
        return class_id

    else:
        # If no class label is found in the LLM text, return the last label.
        return len(label_names) - 1

def evaluate(
    model : AutoModelForCausalLM,
    tokenizer : AutoTokenizer,
    label_names : list,
    eval_dataset : Dataset,
    eval_config : EvaluationConfig
    ) -> EvaluationResult:
    """
    Evaluate an LLM's text classification performance on a supervised dataset.

    Args:
        model (AutoModelForCausalLM): The LLM to use. It can be pre-trained or fine-tuned.
        tokenizer (AutoTokenizer): The tokenizer to use. This should come with the LLM.
        label_names (list): The name of each class label in the evaluation dataset.
        eval_dataset (Dataset): The evaluation dataset. Must be preprocessed (see ``finetune.preprocess_dataset()``).
        eval_config (EvaluationConfig): Controls what instructions to give to the LLM to classify each sample.

    Returns:
        EvaluationResult: Raw evaluation data, including all samples, predicted/actual labels, and the LLM's response for each sample.
    """

    # Add an "I don't know" label to the end of the label names list.
    # We will need this as a fallback if the LLM does not provide a
    # class label in its answer.
    label_names.append("Unknown")

    labels_pred = []
    llm_responses = []

    # Get all text inputs (X) in eval_dataset
    texts = [message[0]['content'].strip() for message in eval_dataset['messages']]

    # For every sample:
    for text in tqdm(texts, "Evaluating model"):
        
        # Generate a classification prompt for the sample
        prompt = [
            {"role":"system", "content":eval_config.prompt},
            {"role":"user", "content":text}
        ]
        # Remove the system prompt from the chat template if none was specified
        if eval_config.prompt is None or eval_config.prompt == "":
            prompt.pop(0)

        # Get the LLM to generate an answer
        response = ft.generate(
                            prompt=prompt, model=model, tokenizer=tokenizer,
                            max_new_tokens = eval_config.max_tokens
                            )
        
        # Extract the class ID from the LLM's answer if one exists
        pred_class = _get_class_id_from_model_response(response, label_names)

        labels_pred.append(pred_class)
        llm_responses.append(response)

    # Get all class label IDs (y_true) in eval_dataset
    groundtruth = [message[-1]['content'] for message in eval_dataset['messages']]
    labels_true = [_get_class_id_from_model_response(label, label_names) for label in groundtruth]

    return EvaluationResult(
        config=eval_config,
        texts=texts,
        labels_pred=labels_pred,
        labels_true=labels_true,
        label_names=label_names,
        llm_responses=llm_responses)
