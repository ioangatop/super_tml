import torch
import argparse


# ----- Print Parser -----

def print_args(ARGS):
    print('\n'+26*'='+' Configuration '+26*'=')
    for name, var in vars(ARGS).items():
        print('{} : {}'.format(name, var))
    print('\n'+25*'='+' Training Starts '+25*'='+'\n')


# ----- Parser -----

def parser():
    PARSER = argparse.ArgumentParser(description='Training parameters.')

    PARSER.add_argument('--dataset', default='iris', type=str,
                        choices=['iris', 'wine'], help='Dataset.')
    PARSER.add_argument('--model', default='densenet121', type=str,
                        choices=['resnet18', 'densenet121'], help='Model.')
    PARSER.add_argument('--opt', default='Adagrad', type=str,
                        choices=['Adam', 'Adamax', 'Adagrad'], help='Optimizer.')

    PARSER.add_argument('--epochs', default=10, type=int,
                        help='Number of training epochs.')
    PARSER.add_argument('--batch_size', default=16, type=int,
                        help='Batch size.')
    PARSER.add_argument('--val_size', default=0.2, type=int,
                        help='Validation size. Proportion of the train dataset.')
    PARSER.add_argument('--test_size', default=0.2, type=int,
                        help='Test size proportion of the hole dataset.')

    PARSER.add_argument('--seed', default=0, type=int,
                        help='Fix random seed.')
    PARSER.add_argument('--tags', default='logs', type=str,
                        help='Run tags.')
    PARSER.add_argument('--device', default=None, type=str,
                        choices=['cpu', 'cuda'],
                        help='Device to run the experiment.')

    ARGS = PARSER.parse_args()

    if ARGS.device is None:
        ARGS.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print_args(ARGS)

    return ARGS


args = parser()

if __name__ == "__main__":
    pass
