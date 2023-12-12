# Notice that the "adapt_steps" and "iters" stand for the inner and outer iterations in this code.
import os
import random
import numpy as np
import torch
from torch import nn
import learn2learn as l2l
import pickle
import argparse
import datetime

class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)
 
def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch,
               learner,
               features,
               loss,
               reg_lambda,
               adaptation_steps,
               shots,
               ways,
               device=None):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    data = features(data)
    # Separate data into adaptation/evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    train_error = loss(learner(adaptation_data), adaptation_labels)
    learner.adapt(train_error)

    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        ways=5,
        shots=1,
        meta_lr=0.0005,
        fast_lr=0.0005,
        reg_lambda=0,
        adapt_steps=1,  
        meta_bsz=32,
        iters=2000,  
        cuda=1,
        seed=42,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device(cuda)


    tasksets = l2l.vision.benchmarks.get_tasksets('omniglot',
                                                  train_ways=ways,
                                                  train_samples=2 * shots,
                                                  test_ways=ways,
                                                  test_samples=2 * shots,
                                                  num_tasks=2000,
                                                  root='~/data',
                                                  )

    # Create model
    features = l2l.vision.models.OmniglotFC(28 ** 2, ways).features
    features.to(device)
    head = l2l.vision.models.OmniglotFC(28 ** 2, ways).classifier
    head = l2l.algorithms.MAML(head, lr=fast_lr)
    head.to(device)

    # Setup optimization
    all_parameters = list(features.parameters())

    ## use different learning rates for w and theta
    optimizer = torch.optim.Adam(all_parameters, lr=meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    training_error = torch.ones(iters)
    test_error = torch.ones(iters)
    val_error = torch.ones(iters)

    train_acc = torch.ones(iters)
    test_acc = torch.ones(iters)
    val_acc = torch.ones(iters)

    running_time = np.ones(iters)
    import time
    start_time = time.time()

    for iteration in range(iters):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        for task in range(meta_bsz):
            # Compute meta-training loss
            learner = head.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               features,
                                                               loss,
                                                               reg_lambda,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = head.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               features,
                                                               loss,
                                                               reg_lambda,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            learner = head.clone()
            batch = tasksets.test.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               features,
                                                               loss,
                                                               reg_lambda,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        training_error[iteration] = meta_train_error / meta_bsz
        test_error[iteration] = meta_test_error / meta_bsz
        val_error[iteration] = meta_valid_error / meta_bsz

        train_acc[iteration] = meta_train_accuracy / meta_bsz
        test_acc[iteration] = meta_test_accuracy / meta_bsz
        val_acc[iteration] = meta_valid_accuracy / meta_bsz

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_bsz)
        print('Meta Train Accuracy', meta_train_accuracy / meta_bsz)
        print('Meta Valid Error', meta_valid_error / meta_bsz)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_bsz)
        print('Meta Test Error', meta_test_error / meta_bsz)
        print('Meta Test Accuracy', meta_test_accuracy / meta_bsz)

        # Average the accumulated gradients and optimize
        for p in all_parameters:
            p.grad.data.mul_(1.0 / meta_bsz)

        optimizer.step()
        end_time = time.time()
        running_time[iteration] = end_time - start_time
        print('total running time', end_time - start_time)

    return training_error.numpy(), test_error.numpy(), val_error.numpy(), \
           train_acc.numpy(), test_acc.numpy(), val_acc.numpy(), running_time


if __name__ == '__main__':
    train_error = []
    test_error = []
    val_error = []
    train_acc = []
    test_acc = []
    val_acc = []
    run_time = []
    
    algorithm = "SSGD"
    # algorithm = "TSGD"
    inner_loops = 32
    parser = argparse.ArgumentParser()
    parser.add_argument('--inner_step', default=inner_loops, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    args = parser.parse_args()
    seeds = [42, 52, 62, 72, 82]
    if algorithm =='SSGD':
        stp = 0
    else:
        stp = args.inner_step
    lr = 0.0005
    fastlr = 0.0005
    reg = 0
    device = args.device
    for seed in seeds:
        training_error, testing_error, valid_error, train_a, test_a, val_a, running_time = main(meta_lr=lr,
                                                                 adapt_steps=stp,
                                                                 fast_lr=fastlr,
                                                                 reg_lambda=reg,
                                                                 iters=2000,
                                                                 seed=seed,
                                                                 cuda=device)
        train_error.append(training_error)
        test_error.append(testing_error)
        val_error.append(valid_error)

        train_acc.append(train_a)
        test_acc.append(test_a)
        val_acc.append(val_a)

        run_time.append(running_time)

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if not os.path.exists('./exp_data/' + now):
        os.mkdir('./exp_data/' + now)
    pstr = 'omniglot_lr_' + str(lr) + '_fastlr_' + str(fastlr) + '_steps_' + str(stp)
    with open('./exp_data/' + now + '/train_loss' + pstr, 'wb+') as f:
        pickle.dump(train_error, f)
    with open('./exp_data/' + now + '/test_loss' + pstr, 'wb+') as f:
        pickle.dump(test_error, f)
    with open('./exp_data/' + now + '/valid_loss' + pstr, 'wb+') as f:
        pickle.dump(val_error, f)
    with open('./exp_data/' + now + '/train_accuracy' + pstr, 'wb+') as f:
        pickle.dump(train_acc, f)
    with open('./exp_data/' + now + '/valid_accuracy' + pstr, 'wb+') as f:
        pickle.dump(val_acc, f)
    with open('./exp_data/' + now + '/test_accuracy' + pstr, 'wb+') as f:
        pickle.dump(test_acc, f)
    with open('./exp_data/' + now + '/run_time' + pstr, 'wb+') as f:
        pickle.dump(run_time, f)
