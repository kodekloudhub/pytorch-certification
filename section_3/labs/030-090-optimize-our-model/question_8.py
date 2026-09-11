"""
Complete the training helper and compare SGD, Adam, and AdamW on the same regression problem. Print the final loss for each optimizer.

NOTE: Create a fresh model inside the helper so every optimizer starts with identical weights. Each training iteration must clear gradients, calculate MSE loss, run backward, and update the parameters.
"""
from torch import nn, optim
from pre import create_regression_model, regression_inputs, regression_labels


def train_with(optimizer_type, learning_rate, weight_decay=0.0):
    # Create a new model for each comparison so every optimizer starts with
    # the same initial parameters.
    model = ____()

    # optimizer_type will be SGD, Adam, or AdamW. Configure it with the model's
    # parameters and the values passed into this function.
    optimizer = optimizer_type(
        model.parameters(),
        lr=____,
        weight_decay=____,
    )

    # Use the regression loss function demonstrated earlier in the lab.
    criterion = nn.____()

    # Train each model for the same number of iterations to make the comparison
    # consistent.
    for _ in range(100):
        # Step 1: Clear gradients left over from the previous iteration.
        optimizer.____()

        # Step 2: Run the model and calculate the loss.
        predictions = ____(regression_inputs)
        loss = ____(predictions, ____)

        # Step 3: Calculate gradients and update the model parameters.
        loss.____()
        optimizer.____()

    # Return a regular Python number so it is easy to display and compare.
    return loss.item()


# Run the same helper with each optimizer. The learning rates are configured
# independently because optimizers do not necessarily use the same best rate.
optimizer_results = {
    "SGD": train_with(optim.____, learning_rate=0.1),
    "Adam": train_with(optim.____, learning_rate=0.05),
    # AdamW also demonstrates decoupled weight decay.
    "AdamW": train_with(
        optim.____,
        learning_rate=0.05,
        weight_decay=0.01,
    ),
}

# Display the final training loss produced by each optimizer.
for optimizer_name, final_loss in optimizer_results.items():
    print(f"{optimizer_name:5} final loss: {final_loss:.6f}")
