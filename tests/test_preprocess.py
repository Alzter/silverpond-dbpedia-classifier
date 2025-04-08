import pytest
import finetune as ft
from datasets import load_dataset, ClassLabel
import numpy as np

@pytest.fixture()
def full_dataset():
    return load_dataset("fancyzhx/dbpedia_14")

@pytest.fixture
def test_dataset():
    return load_dataset("fancyzhx/dbpedia_14", split="test")

def test_resize_datadict(full_dataset):
    size = np.array(list(full_dataset.shape.values()))[:,0]

    sample = ft.sample_dataset(full_dataset, ratio=0.5)

    new_size = np.array(list(sample.shape.values()))[:,0]

    assert np.array_equal(size // 2, new_size)

def test_resize_dataset(test_dataset):
    size = len(test_dataset)

    sample_ratio = ft.sample_dataset(test_dataset, ratio=0.5)
    assert len(sample_ratio) == size // 2

    sample_size = ft.sample_dataset(test_dataset, size=1400)
    assert len(sample_size) == 1400

    sample_per_class = ft.sample_dataset(test_dataset, samples_per_class=10)
    assert len(sample_per_class) == 14 * 10

def test_resize_dataset_requires_method(test_dataset):
    with pytest.raises(ValueError):
        ft.sample_dataset(test_dataset)

def test_resize_dataset_has_no_side_effects(test_dataset):
    expected = len(test_dataset)

    ft.sample_dataset(test_dataset, ratio=0.1)

    actual = len(test_dataset)

    assert expected == actual, "There should be no side effects from resizing a dataset"

def test_preprocess_datadict(full_dataset):
    subsets = ["train", "test"]
    full_dataset = ft.sample_dataset(full_dataset, samples_per_class=10)
    processed_datadict, _ = ft.preprocess_dataset(full_dataset, "content", "label")

    for subset in subsets:
        assert processed_datadict[subset].column_names == ["messages"]

def test_preprocess_dataset(test_dataset):
    samples = test_dataset['content']
    labels = test_dataset['label']
    
    # Translate label IDs to names
    if type(test_dataset.features['label']) is ClassLabel:
        label_names = test_dataset.features['label'].names
        labels = [label_names[label] for label in labels]

    processed_dataset, _ = ft.preprocess_dataset(test_dataset, "content", "label")

    assert processed_dataset.column_names == ["messages"], "Dataset must be in conversational format"

    assert samples == [message[0]['content'] for message in processed_dataset['messages']], "The content of the pre-processed dataset should be identical"
    assert labels == [message[-1]['content'] for message in processed_dataset['messages']], "The content of the pre-processed dataset should be identical"

def test_preprocess_dataset_no_side_effects(test_dataset):
    expected = test_dataset.column_names

    ft.preprocess_dataset(test_dataset, "content", "label")

    actual = test_dataset.column_names

    assert expected == actual, "There should be no side effects from resizing a dataset"

def test_get_class_labels(test_dataset):

    expected = test_dataset.features['label'].names
    _, actual = ft.preprocess_dataset(test_dataset, "content", "label")

    assert expected == actual, "A list of class label names should be returned when preprocessing a dataset."