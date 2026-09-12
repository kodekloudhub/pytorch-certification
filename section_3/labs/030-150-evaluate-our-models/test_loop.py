"""
Utilize Accuracy from torchmetrics to compute accuracy.

Set the model to evaluation mode and use torch.inference_mode() while running
the test loop.

NOTE: model.eval() changes evaluation-sensitive layer behavior, while
torch.inference_mode() disables gradient tracking and additional autograd
bookkeeping for inference-only work. Reset a stateful metric after computing it
so a later evaluation starts with empty metric state.
"""
from load_data import test_loader
# Import modules
import ____
import torch

# Initialize the accuracy metric as a multiclass task and set number of classes to 2
accuracy_metric = ____.____(____=____, ____=____)

# Function for evaluating a model
def evaluate_model(model):

    # Set model to evaluation mode
    ____.____
    # Use inference mode because this test pass is inference-only
    with ____.____:
        # Loop over the test dataloader
        for i, data in enumerate(____, 0):
            inputs, labels = data
            outputs = model(inputs)

            # Get the class with the highest logit for each input
            predicted = outputs.argmax(dim=1)

            # Update the accuracy metric with predictions and true labels
            accuracy_metric.update(predicted, ____)

    # Compute the final accuracy
    final_accuracy = accuracy_metric.____
    print(f"Accuracy: {final_accuracy * 100}%")

    # Reset the metric before evaluating another model
    accuracy_metric.reset()
