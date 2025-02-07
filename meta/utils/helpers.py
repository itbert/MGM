import torch
import torch.nn as nn
import torch.optim as optim


def create_optimizers(models, learning_rate=0.001):
    optimizers = {}
    for name, model in models.items():
        optimizers[name] = optim.Adam(model.parameters(), lr=learning_rate)
        
    return optimizers


def train_step(
    models, optimizers, criterion, task_description,
    true_architecture, true_weights, true_quality_metric
    ):
    encoder, architecture_generator, weight_generator, evaluator = models.values()
    
    encoded_task = encoder(task_description)
    generated_architecture = architecture_generator(encoded_task)
    generated_weights = weight_generator(encoded_task)
    quality_metric = evaluator(generated_architecture)

    loss_architecture = criterion['architecture'](generated_architecture, 
                                                  true_architecture)
    loss_weights = criterion['weights'](generated_weights, 
                                        true_weights)
    loss_evaluator = criterion['evaluator'](quality_metric, 
                                            true_quality_metric)

    optimizers['encoder'].zero_grad()
    optimizers['architecture_generator'].zero_grad()
    optimizers['weight_generator'].zero_grad()
    optimizers['evaluator'].zero_grad()

    loss_architecture.backward(retain_graph=True)
    loss_weights.backward(retain_graph=True)
    loss_evaluator.backward()

    optimizers['encoder'].step()
    optimizers['architecture_generator'].step()
    optimizers['weight_generator'].step()
    optimizers['evaluator'].step()

    return loss_architecture.item(), loss_weights.item(), loss_evaluator.item()


def create_model_from_architecture(architecture, num_classes):
    layers = []
    for i, layer_params in enumerate(architecture):
        if i == 0:
            layers.append(nn.Conv2d(3, 
                                    int(layer_params[0]), 
                                    kernel_size=3, 
                                    padding=1))
        else:
            layers.append(nn.Conv2d(int(architecture[i-1][0]), 
                                    int(layer_params[0]), 
                                    kernel_size=3, 
                                    padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(int(architecture[-1][0]), num_classes))
    return nn.Sequential(*layers)


def train_generated_model(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Generated Model Epoch [{epoch+1}/{num_epochs}], 
              Loss: {avg_loss:.4f}')


def train_model(models, optimizers, criterion, dataloader, num_epochs=10):
    for epoch in range(num_epochs):
        total_loss_architecture = 0.0
        total_loss_weights = 0.0
        total_loss_evaluator = 0.0

        for (task_description, 
             true_architecture, 
             true_weights, 
             true_quality_metric) in dataloader:
            loss_architecture, loss_weights, loss_evaluator = train_step(
                models, optimizers, criterion, 
                task_description, 
                true_architecture, true_weights, true_quality_metric)

            total_loss_architecture += loss_architecture
            total_loss_weights += loss_weights
            total_loss_evaluator += loss_evaluator

        avg_loss_architecture = total_loss_architecture / len(dataloader)
        avg_loss_weights = total_loss_weights / len(dataloader)
        avg_loss_evaluator = total_loss_evaluator / len(dataloader)

        print(f'Epoch [{epoch+1}/{num_epochs}], 
              Loss Architecture: {avg_loss_architecture:.4f}, '
              f'Loss Weights: {avg_loss_weights:.4f}, 
              Loss Evaluator: {avg_loss_evaluator:.4f}')
